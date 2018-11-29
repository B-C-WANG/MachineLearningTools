from distutils.core import setup


package_dir = {"MLT.Classification":"MLT/Classification",
               "MLT.DataPreprocessing": "MLT/DataPreprocessing",
               "MLT.Regression": "MLT/Regression",
               "MLT.VAE":"MLT/VAE"}

packages = [
      "MLT.Classification",
"MLT.DataPreprocessing",
"MLT.Regression",
      "MLT.VAE"

]




setup(
      name="MLT",
      version="0.1",
      description="Machine Learning Tools",
      author="B.C.WANG",
      url="https://github.com/B-C-WANG",
      license="LICENSE",
      package_dir=package_dir,
      packages=packages
      )