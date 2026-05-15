from setuptools import setup, find_packages

setup(
    name="credit_fraud_ml",
    version="1.0.0",
    description="Credit Card Fraud Detection ML Framework",
    author="Kgothatso Ntumbe",
    author_email="kgothatso@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "fastapi>=0.104.0",
        "pydantic>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fraud-train=models.train:main",
            "fraud-predict=deployment.predictor:main",
            "fraud-api=deployment.api:main",
        ]
    },
)
