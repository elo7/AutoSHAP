from autoshap.version import __version__
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import xgboost
import shap
import os
from datetime import date
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib as mpl
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display

plt.rc('axes', axisbelow=True)
mpl.rc('figure', max_open_warning = 0)

class SHAPPipeline:
    """SHapley Additive Explanations automatization
    with xgboost model

    Parameters
    ----------
    ml_model_params : dict
        Contains all the following parameters

    binary_output_feature : str
        Name of a binary variable which will predicted by the machine learning model

    dt_begin : str, yyyy-MM-dd
        Date to be used to compute Shapley Values

    dt_end : str, yyy-MM-dd
        Not implemented yet

    train_samples : int
        Number of observations used in the model

    shap_calculates_samples : int
        Number of observations used to compute Shapley Values. Must be less
        or equal to 'train_samples'

    interaction : bool, default = False
        If set to True the model can compute the Shapley Interaction Values.
        This can take extremely long times for larger datasets, use with caution

    tuning_params : dict
        Dictionary containing the lists of values for hyperparameter optimization.
        Must have a list for each of the following xgboost parameters:
        'learning_rate', 'max_depth' and 'min_child_weight'
    """

    def __init__(self, ml_model_params):
        self.binary_output_feature = ml_model_params['binary_output_feature']
        self.dt_begin = ml_model_params.get('dt_begin', None)
        self.dt_end = ml_model_params.get('dt_end', None)
        self.train_samples = ml_model_params['train_samples']
        self.shap_calculate_samples = int(ml_model_params['shap_calculate_samples'])
        self.interaction = ml_model_params['interaction']
        self.tuning_params = ml_model_params['tuning_params']
        self.best_params = {'max_depth':6,
                            'min_child_weight': 1,
                            'learning_rate':0.1,
                            'objective':'reg:squarederror'}
        self.path_to_data = ml_model_params['path_to_data']
        self.best_model = None
        self.shap_interaction_values = None
        self.verbose = ml_model_params['verbose']

    def validate_dt_col(self, df):
        try:
            df['dt']
            return df
        except:
            current_date = str(date.today())
            if self.verbose == 1:
                print(f'No "dt" column in dataframe. Setting it as {current_date}')
            df['dt'] = [current_date]*df.shape[0]
            self.dt_begin = current_date
            return df

    def validate_dataset_balance(self, df):
        df = self.validate_dt_col(df)
        min_count = df[df['dt']==self.dt_begin] \
            .groupby(by=self.binary_output_feature)[self.binary_output_feature] \
            .value_counts() \
            .values \
            .min()
        if int(self.train_samples /2) > min_count:
            if self.verbose == 1:
                print(f'Chosen n_samples ({self.train_samples}) for dataset has insufficient counts for one of the classes.')
                print(f'n_samples will be set as {int(min_count*2)}')
            return False, int(2*min_count)
        else:
            return True, None

    def set_n_samples(self, df):
        balanced, n_samples = self.validate_dataset_balance(df)
        if balanced == False:
            return n_samples
        else:
            return self.train_samples

    def transform_dataset(self, df):
        self.train_samples = self.set_n_samples(df)
        sampled_df = df[
            (df[self.binary_output_feature]==0)&\
            (df['dt']==self.dt_begin)] \
                .sample(int(self.train_samples / 2))
        sampled_df = pd.concat(
            [sampled_df, df[df[self.binary_output_feature]==1] \
             .sample(int(self.train_samples / 2))])
        sampled_df = sampled_df.sample(frac=1.0, random_state=None)
        sampled_df = sampled_df.drop('dt', axis=1)
        sampled_df.reset_index(drop=True, inplace=True)
        feature_names = list(sampled_df.drop(self.binary_output_feature, axis=1).columns)
        return feature_names, sampled_df

    def set_data(self, feature_names, sampled_df):
        X = sampled_df[feature_names].copy()
        y = sampled_df[self.binary_output_feature].copy()
        dtrain = xgboost.DMatrix(X, label=y.values)
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.dtrain = dtrain
        return self.X, self.y, self.dtrain

    def train(self):
        params = {
        'max_depth':6,
        'min_child_weight': 1,
        'learning_rate':0.1,
        'objective':'reg:squarederror'}

        min_accuracy = 0.0
        best_params = [None, None, None]
        for max_depth in self.tuning_params['max_depth']:
            for min_child_weight in self.tuning_params['min_child_weight']:
                for learning_rate in self.tuning_params['learning_rate']:
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['learning_rate'] = learning_rate

                    model = xgboost.train(params=params, dtrain=self.dtrain,
                                    evals=[(self.dtrain, 'test'), (self.dtrain, 'train')], early_stopping_rounds=2000, verbose_eval=0)
                    y_pred = np.round(model.predict(self.dtrain), 0)

                    acc = accuracy_score(y_pred, self.y)
                    if acc > min_accuracy:
                        min_accuracy = acc
                        best_params = (max_depth, min_child_weight, learning_rate)

        self.best_params = {"max_depth":best_params[0], "min_child_weight":best_params[1], "learning_rate":best_params[2]}
        return self.best_params

    def set_best_model(self):
        self.best_model = xgboost.train(params=self.best_params, dtrain=self.dtrain, evals=[(self.dtrain, 'test'), (self.dtrain, 'train')],
                              early_stopping_rounds=1000, verbose_eval=0)
        return self.best_model

    def best_model_log(self):
        class_report = classification_report(self.y_pred, self.y.values)
        curr_date = date.today()
        text_file = open(f'{self.path_to_data}/train_log_{curr_date}.txt', "a")
        text_file.write("########################################\n"\
                    f"log from {curr_date}\n"\
                    f"trained with          : {self.best_model.__class__}\n"\
                    f"parameters            : {self.best_params}\n"\
                    f"train_samples         : {self.train_samples}\n"\
                    f"shap_calculate_samples: {self.shap_calculate_samples}\n"\
                    f"data date             : {self.dt_begin}\n"\
                    "classification report:\n"
                    f"{class_report}\n\n")
        text_file.close()

    def calc_predictions(self):
        self.y_pred = np.round(self.best_model.predict(self.dtrain), 0)

    def select_correct_predictions(self):
        correct_preds_indices = np.where(self.y.values == self.y_pred)[0]
        self.X = self.X.loc[correct_preds_indices].copy()
        self.y = self.y.loc[correct_preds_indices].copy()
        self.y_pred = self.y_pred[correct_preds_indices]

    def save_model(self):
        joblib.dump(self.best_model, f'{self.path_to_data}/xgboost_model_{self.binary_output_feature}.joblib')

    def validate_shap_calculate_samples(self):
        if self.shap_calculate_samples > min(self.train_samples, self.X.shape[0]):
            print(f'shap_calculate_samples is less than train_samples.')
            print(f'Setting shap_calculate_samples = {self.train_samples}')
            return min(self.train_samples, self.X.shape[0])
        else:
            return self.shap_calculate_samples

    def build_shap_df(self):
        self.shap_calculate_samples = self.validate_shap_calculate_samples()
        shap_df = pd.DataFrame(data=self.shap_values, columns=list(self.X.columns))
        shap_df = shap_df.iloc[:self.shap_calculate_samples].copy()
        shap_df[self.binary_output_feature] = self.y_pred[:self.shap_calculate_samples]
        self.shap_df = shap_df

    def explainer(self):
        return shap.TreeExplainer(self.best_model)

    def compute_shap_values(self):
        self.shap_values = self.explainer().shap_values(self.X)
        return self.shap_values

    def compute_shap_interaction_values(self):
        self.shap_interaction_values = self.explainer().shap_interaction_values(self.X)
        return self.shap_interaction_values

    def save_shap_data(self):
        joblib.dump(self.explainer(), f'{self.path_to_data}/model_explainer_{self.binary_output_feature}.joblib')
        np.save(f'{self.path_to_data}/shap_values_{self.binary_output_feature}.npy', self.shap_values[:self.shap_calculate_samples])
        if self.interaction:
            np.save(f'{self.path_to_data}/shap_interaction_values_{self.binary_output_feature}.npy', self.shap_interaction_values[:self.shap_calculate_samples])
        np.save(f'{self.path_to_data}/y_{self.binary_output_feature}.npy', self.y[:self.shap_calculate_samples].values)
        self.X.iloc[:self.shap_calculate_samples].to_csv(f'{self.path_to_data}/X_shap_data_{self.binary_output_feature}.csv', index=False)
        self.shap_df.to_csv(f'{self.path_to_data}/shap_df_data_{self.binary_output_feature}.csv', index=False)

    def plot_shap_values(self, df, title='Shapley Values', figsize=(15, 4),
                     save_name=None, absolute=True, max_display=None, show_plots=True):

        if max_display is None:
            max_display = len(df.columns)

        plt.figure(figsize=figsize)
        plt.title(title)
        if absolute:
            df = df.sort_values(by='shapley_values', ascending=False)
            ax = sns.barplot(x="shapley_values",
                             y="features",
                             data=df[:max_display], color='#1E88E5', ci=95)
            plt.xlabel("Mean (|Shap value|): average impact on model output magnitude", fontsize=12)
            plt.xlim([-0.001, df['shapley_values'].max()*1.05])
        else:
            col = "shapley_values_non_abs"
            if max_display < len(df.features.unique()):
                df['abs_val'] = df[col].abs()
                df = df.sort_values(by='abs_val', ascending=False)[:max_display]
            df = df.sort_values(by=col, ascending=False)
            col_xlabel = "Mean (Shap value)"
            ax = sns.barplot(x="shapley_values_non_abs",
                             y="features",
                             data=df,
                             color='#1E88E5')
            plt.xlabel("Mean (Shap value): average impact on model output magnitude", fontsize=12)
            plt.xlim([df['shapley_values_non_abs'].min()*1.05, df['shapley_values_non_abs'].max()*1.05])

        plt.ylabel("Features", fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        if show_plots == True:
            plt.show()
        if save_name:
            plt.savefig(save_name+".png", bbox_inches='tight',dpi=300)
        plt.close()

    def _make_shap_values_stats(self, shap_values, X):
        shap_values_df = pd.DataFrame(data=shap_values, columns=X.columns)
        shap_values_stats_df = pd.DataFrame(
            {
            "shapley_values":np.mean(abs(shap_values_df), axis=0),
            "shapley_values_std":np.std(abs(shap_values_df), axis=0),
            "shapley_values_non_abs":np.mean(shap_values_df, axis=0),
            "features":shap_values_df.columns})
        return shap_values_stats_df.reset_index(drop=True)

    def make_shap_values_stats(self, shap_values, X, samples):
        a = self._make_shap_values_stats(shap_values, X)
        df = pd.DataFrame(data=[], columns=a.columns)
        for i in range(X.shape[0]//samples):
            df = pd.concat([df, self._make_shap_values_stats(shap_values[i*samples:(i+1)*samples], X[i*samples:(i+1)*samples])])
        return df

    def make_summary_plots(self, shap_values, X, max_display, label="", show_plots=True):
        plt.figure()
        plt.title(f'Summary plot - {self.binary_output_feature}: {label}')
        shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=max_display)
        if show_plots == True:
            plt.show()
        plt.savefig(f'{self.path_to_data}/summary_plot_dots_{self.binary_output_feature}_{label}.png', bbox_inches='tight',dpi=300)
        plt.close()

    def make_summary(self, n_high_contribution_cols = 10, n_high_contribution_interaction_cols = 3,
                  max_display=None, show_plots=True):

        if self.verbose == 1:
            print('Loading data...')

        explainer = joblib.load(f'{self.path_to_data}/model_explainer_{self.binary_output_feature}.joblib')
        model = joblib.load(f'{self.path_to_data}/xgboost_model_{self.binary_output_feature}.joblib')
        shap_values = np.load(f'{self.path_to_data}/shap_values_{self.binary_output_feature}.npy')
        if self.interaction:
            shap_interaction_values = np.load(f'{self.path_to_data}/shap_interaction_values_{self.binary_output_feature}.npy', allow_pickle=True)
        y = np.load(f'{self.path_to_data}/y_{self.binary_output_feature}.npy', allow_pickle=True)
        X = pd.read_csv(f'{self.path_to_data}/X_shap_data_{self.binary_output_feature}.csv')
        shap_df = pd.read_csv(f'{self.path_to_data}/shap_df_data_{self.binary_output_feature}.csv')

        if self.verbose == 1:
            print('Building dataframes...')

        shap_values_df = pd.DataFrame(data=shap_values, columns=X.columns)
        shap_values_df = pd.concat([shap_df[self.binary_output_feature].iloc[:self.shap_calculate_samples].copy(),
                shap_values_df], axis=1)

        list0 = list(shap_df[shap_df[self.binary_output_feature]==0].reset_index(drop=True).index)
        list1 = list(shap_df[shap_df[self.binary_output_feature]==1].reset_index(drop=True).index)
        shap_values_stats_df = self._make_shap_values_stats(shap_values, X)
        shap_values_stats_0_df = self._make_shap_values_stats(shap_values[list0, :], X.iloc[list0])
        shap_values_stats_1_df = self._make_shap_values_stats(shap_values[list1, :], X.iloc[list1])

        if self.verbose == 1:
            print('Making summary plots...')

        self.plot_shap_values(shap_values_stats_df,
                         figsize=(8, 10),
                         absolute=True,
                         save_name=f'{self.path_to_data}/summary_plot_{self.binary_output_feature}_absolute',
                         title=f'Shapley values - {self.binary_output_feature}: absolute', max_display=max_display,
                         show_plots=show_plots)
        self.plot_shap_values(shap_values_stats_df,
                         figsize=(8, 10),
                         absolute=False,
                         save_name=f'{self.path_to_data}/summary_plot_{self.binary_output_feature}_non_absolute',
                         title=f'Shapley values - {self.binary_output_feature}: non absolute', max_display=max_display,
                         show_plots=show_plots)
        self.plot_shap_values(shap_values_stats_0_df,
                         figsize=(8, 10),
                         absolute=True,
                         save_name=f'{self.path_to_data}/summary_plot_{self.binary_output_feature}_absolute_label0',
                         title=f'Shapley values - {self.binary_output_feature}: absolute labels 0', max_display=max_display,
                         show_plots=show_plots)
        self.plot_shap_values(shap_values_stats_1_df,
                         figsize=(8, 10),
                         absolute=True,
                         save_name=f'{self.path_to_data}/summary_plot_{self.binary_output_feature}_absolute_label1',
                         title=f'Shapley values - {self.binary_output_feature}: absolute labels 1', max_display=max_display,
                         show_plots=show_plots)

        self.make_summary_plots(shap_values = shap_values,
                           X = X, max_display=max_display,
                           label="", show_plots=show_plots)

        self.make_summary_plots(shap_values = shap_values[list0],
                           X = X.iloc[list0].copy(), max_display=max_display,
                           label="label0", show_plots=show_plots)

        self.make_summary_plots(shap_values = shap_values[list1],
                           X = X.iloc[list1].copy(), max_display=max_display,
                           label="label1", show_plots=show_plots)

        if self.verbose == 1:
            print('Making dependence plots...')

        high_contribution_cols = list(shap_values_df \
                                      .drop(self.binary_output_feature, axis=1) \
                                      .abs() \
                                      .mean(axis=0) \
                                      .sort_values(ascending=False) \
                                      .keys())[:n_high_contribution_cols]

        for item in high_contribution_cols:
            plt.figure()
            shap.dependence_plot(item, shap_values, X, x_jitter=0.2, show=False, interaction_index=None)
            plt.axhline(y=0.0, color='deeppink', linestyle='--',zorder=0.2)
            plt.grid(True)
            plt.savefig(f'{self.path_to_data}/dependence_{self.binary_output_feature}_{item}.png', bbox_inches='tight',dpi=300)
            if show_plots == True:
                plt.show();
            plt.close();

        if self.interaction:
            if self.verbose == 1:
                print('Making dependence interaction plots...')

            high_contribution_cols = list(shap_values_df \
                                          .drop(self.binary_output_feature, axis=1) \
                                          .abs() \
                                          .mean(axis=0) \
                                          .sort_values(ascending=False) \
                                          .keys())[:n_high_contribution_interaction_cols]

            for item in itertools.combinations(high_contribution_cols, 2):
                plt.figure()
                shap.dependence_plot((item[0], item[1]), shap_interaction_values, X, x_jitter=0.2, show=False)
                plt.axhline(y=0.0, color='deeppink', linestyle='--',zorder=0.2)
                plt.grid(True)
                plt.savefig(f'{self.path_to_data}/dependence_{self.binary_output_feature}_{item[0]}_{item[1]}.png', bbox_inches='tight',dpi=300)
                if show_plots == True:
                    plt.show();
                plt.close();

        if self.verbose == 1:
            print('We are done!')

class FullPipeline(SHAPPipeline):
    def __init__(self, ml_model_params):
        super().__init__(ml_model_params)

    def run_full_pipeline(self, df):
        feature_names, sampled_df = self.transform_dataset(df=df)
        _, _, _ = self.set_data(feature_names, sampled_df)
        self.train()
        self.set_best_model()
        self.calc_predictions()
        self.best_model_log()
        self.select_correct_predictions()
        _ = self.compute_shap_values()
        self.build_shap_df()
        if self.interaction:
            _ = self.compute_shap_interaction_values()
        self.save_shap_data()
        self.save_model()

class SHAPViews:
    def __init__(self,shap_df,data_df,interaction_values=None):
        self.shap_df = shap_df
        self.data_df = data_df
        self.interaction_values = interaction_values
        self.features_names = list(data_df.columns)
        self.feature2id = {v:k for k,v in enumerate(list(data_df.columns))}
    
    def _view_dependence_plot(self,col,width,height,alpha,s,c):
        plt.figure(figsize=(width,height))
        plt.scatter(x=self.data_df[col].values,y=self.shap_df[col].values,marker="." , c=c,alpha=alpha, s=s)
        plt.xlabel(col,fontsize=12)
        plt.ylabel(f"SHAP value for\n{col}",fontsize=12)
        plt.grid(True)
        plt.show()
        
    def view_dependence_plot(self):
        feature_choice = widgets.Dropdown(options=self.features_names,description='Feature:')
        colors = widgets.Dropdown(options=['darkorchid','cornflowerblue','limegreen','darksalmon','black'],description='Color:')
        width = widgets.IntSlider(description="Width", min=2, max=25, value=10, continuous_update=True)
        height = widgets.IntSlider(description="Height", min=2, max=10, value=4, continuous_update=True)
        alpha = widgets.FloatSlider(description="Alpha", min=0, max=1, value=0.7, continuous_update=False)
        s = widgets.FloatSlider(description="Size", min=20, max=800, value=200, continuous_update=False)
        out = widgets.interactive_output(self._view_dependence_plot, {'col': feature_choice, 
                                                                      'width':width, 
                                                                      'height':height, 
                                                                      'alpha':alpha, 
                                                                      's':s,
                                                                      'c':colors})
        feature_choice.layout.width, colors.layout.width, width.layout.width = ['400px','200px','300px']
        height.layout.width, alpha.layout.width, s.layout.width = ['300px']*3
        display(widgets.HBox([feature_choice,colors]),widgets.HBox([width,height]),widgets.HBox([alpha,s]),out)
        
    def _view_dependence_plot_extra(self,col1,col2,width,height,alpha,s):
        col1_id, col2_id = [self.feature2id[col1],self.feature2id[col2]]
        fig, ax = plt.subplots(figsize=(width,height),ncols=1)
        pos = ax.scatter(x=self.data_df[col1].values,
                         y=self.shap_df[col1].values,
                         marker=".", c=self.data_df[col2].values ,alpha=alpha, s=s,cmap=shap.plots.colors.red_blue)
        plt.xlabel(col1,fontsize=12)
        plt.ylabel(f"SHAP value for\n{col1}",fontsize=12)
        ax.set_axisbelow(True)
        plt.grid(True)
        fig.colorbar(pos,ax=ax,label=f'{col2}')
    
    def view_dependence_plot_extra(self):
        col1 = widgets.Dropdown(options=self.features_names,description='Feature 1:', value=self.features_names[0])
        col2 = widgets.Dropdown(options=self.features_names,description='Feature 2:', value=self.features_names[1])
        width = widgets.IntSlider(description="Width", min=2, max=25, value=12, continuous_update=True)
        height = widgets.IntSlider(description="Height", min=2, max=10, value=4, continuous_update=True)
        alpha = widgets.FloatSlider(description="Alpha", min=0, max=1, value=0.7, continuous_update=False)
        s = widgets.FloatSlider(description="Size", min=20, max=800, value=200, continuous_update=False)
        out = widgets.interactive_output(self._view_dependence_plot_extra, {'col1': col1, 'col2':col2, 
                                                                      'width':width, 
                                                                      'height':height, 
                                                                      'alpha':alpha, 
                                                                      's':s})
        col1.layout.width, col2.layout.width, width.layout.width = ['400px','400px','300px']
        height.layout.width, alpha.layout.width, s.layout.width = ['300px']*3
        display(widgets.HBox([col1,col2]),widgets.HBox([width,height]),widgets.HBox([alpha,s]),out)        
        
    def _view_interaction_plot(self,col1,col2,width,height,alpha,s):
        col1_id, col2_id = [self.feature2id[col1],self.feature2id[col2]]
        fig, ax = plt.subplots(figsize=(width,height),ncols=1)
        pos = ax.scatter(x=self.data_df[col1].values,
                         y=self.interaction_values[:,col1_id,col2_id],
                         marker=".", c=self.data_df[col2].values ,alpha=alpha, s=s,cmap=shap.plots.colors.red_blue)
        plt.xlabel(col1,fontsize=12)
        plt.ylabel(f"SHAP interaction value for\n{col1} and {col2}",fontsize=12)
        ax.set_axisbelow(True)
        plt.grid(True)
        fig.colorbar(pos,ax=ax,label=f'{col2}')
    
    def view_interaction_plot(self):
        col1 = widgets.Dropdown(options=self.features_names,description='Feature 1:', value=self.features_names[0])
        col2 = widgets.Dropdown(options=self.features_names,description='Feature 2:', value=self.features_names[1])
        width = widgets.IntSlider(description="Width", min=2, max=25, value=12, continuous_update=True)
        height = widgets.IntSlider(description="Height", min=2, max=10, value=4, continuous_update=True)
        alpha = widgets.FloatSlider(description="Alpha", min=0, max=1, value=0.7, continuous_update=False)
        s = widgets.FloatSlider(description="Size", min=20, max=800, value=200, continuous_update=False)
        out = widgets.interactive_output(self._view_interaction_plot, {'col1': col1, 'col2':col2, 
                                                                      'width':width, 
                                                                      'height':height, 
                                                                      'alpha':alpha, 
                                                                      's':s})
        col1.layout.width, col2.layout.width, width.layout.width = ['400px','400px','300px']
        height.layout.width, alpha.layout.width, s.layout.width = ['300px']*3
        display(widgets.HBox([col1,col2]),widgets.HBox([width,height]),widgets.HBox([alpha,s]),out)
        
    def view_interaction_plot(self):
        col1 = widgets.Dropdown(options=self.features_names,description='Feature 1:', value=self.features_names[0])
        col2 = widgets.Dropdown(options=self.features_names,description='Feature 2:', value=self.features_names[1])
        width = widgets.IntSlider(description="Width", min=2, max=25, value=12, continuous_update=True)
        height = widgets.IntSlider(description="Height", min=2, max=10, value=4, continuous_update=True)
        alpha = widgets.FloatSlider(description="Alpha", min=0, max=1, value=0.7, continuous_update=True)
        s = widgets.FloatSlider(description="Size", min=20, max=800, value=200, continuous_update=True)
        out = widgets.interactive_output(self._view_interaction_plot, {'col1': col1, 'col2':col2, 
                                                                      'width':width, 
                                                                      'height':height, 
                                                                      'alpha':alpha, 
                                                                      's':s})
        col1.layout.width, col2.layout.width, width.layout.width = ['400px','400px','300px']
        height.layout.width, alpha.layout.width, s.layout.width = ['300px']*3
        display(widgets.HBox([col1,col2]),widgets.HBox([width,height]),widgets.HBox([alpha,s]),out)
        
    def _view_dependence_plot_filter(self,col1,col2,width,height,lower,higher,color,x_upper,x_lower,y_upper):
        filtered_df = self.filter_dataframe(col2,lower,higher)
        plt.figure(figsize=(width,height))
        sns.distplot(filtered_df[col1],label=f'Count: {filtered_df.shape[0]}',kde_kws={"lw": 3},
                    color=color, hist=True, hist_kws={"alpha":0.7,"linewidth": 3, "histtype":'stepfilled'})
        sns.distplot(self.data_df[col1],kde_kws={"lw": 3},
                    color='slategray',hist=True, hist_kws={"alpha":0.4,"linewidth": 3, "histtype":'stepfilled'})
        plt.legend(loc=0)
        plt.xlim(self.data_df[col1].min(),self.data_df[col1].max())
        if x_upper != None:
            plt.xlim(None,x_upper)
        plt.ylim(0,y_upper)
        plt.xlabel(col1,fontsize=12)
        plt.grid(True)
        plt.show()
        
    def filter_dataframe(self,col,lower,higher):
        filtered_ids = list(self.shap_df[(self.shap_df[col]>=lower)&(self.shap_df[col]<=higher)].index)
        filtered_df = self.data_df.iloc[filtered_ids].copy()
        return filtered_df
        
    def filter_by_shap_values(self):
        col1 = widgets.Dropdown(options=self.features_names,description='Feature:', value=self.features_names[0])
        col2 = widgets.Dropdown(options=self.features_names,description='Filter Feature:', value=self.features_names[1])
        color = widgets.Dropdown(options=['blueviolet','cornflowerblue','limegreen','darksalmon','black'],description='Color:')
        lower = widgets.FloatSlider(description="Lower limit", min=-1., max=+1., value=-1., step=0.01, continuous_update=True)
        higher = widgets.FloatSlider(description="Upper limit", min=-1., max=1., value=+1., step=0.01, continuous_update=True)
        y_upper = widgets.FloatText(description="Y upper bound",  value=None,step=0.1, continuous_update=True)
        x_upper = widgets.FloatText(description="X upper bound",  value=None,step=0.1, continuous_update=True)
        x_lower = widgets.FloatText(description="X lower bound",  value=None,step=0.1, continuous_update=True)
        width = widgets.IntSlider(description="Width", min=2, max=25, value=12, continuous_update=True)
        height = widgets.IntSlider(description="Height", min=2, max=10, value=4, continuous_update=True)
        out = widgets.interactive_output(self._view_dependence_plot_filter, {'col1': col1, 'col2':col2, 
                                                                      'width':width, 
                                                                      'height':height, 
                                                                      'lower':lower,
                                                                      'higher':higher,
                                                                      'color':color,
                                                                      'y_upper':y_upper,
                                                                      'x_upper':x_upper,'x_lower':x_lower})
        col1.layout.width, col2.layout.width, width.layout.width = ['300px','300px','300px']
        color.layout.width, x_upper.layout.width,x_lower.layout.width, y_upper.layout.width = ["300px","100px","100px","100px"]
        height.layout.width, lower.layout.width, higher.layout.width = ['300px','500px','500px']
        display(widgets.HBox([col1,col2,color]),widgets.HBox([lower,higher]),widgets.HBox([width,height,y_upper]),widgets.HBox([x_lower,x_upper]),out)