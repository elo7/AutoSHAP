from setuptools import setup

version = "0.0.4"

requirements = ["xgboost==0.90",
                "shap==0.31.0",
                "numpy==1.17.3",
                "pandas==0.25.2",
                "scikit-learn==0.21.3",
                "joblib==0.14.0",
                "seaborn==0.9.0"
                ]

packages = ['autoshap']

setup(name='autoshap',
      version=version,
      description='SHapley Additive Explanations automatization with xgboost model',
      url='https://github.com/elo7/AutoSHAP',
      author='Hightower',
      packages=packages,
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)
