This repository contains the "Solar Energy Output Forecaster," a Streamlit-based web application designed to predict solar photovoltaic (PV) energy output using an Artificial Neural Network (ANN). The project integrates real-time and synthetic solar irradiance data, geographical coordinates, and PV system parameters (tilt, azimuth, and kWp) to provide accurate forecasts. Key features include:

- **Data Handling**: Supports NSRDB (National Solar Radiation Database) datasets (e.g., `nsrdb_pretoria.csv`) for Pretoria, South Africa, with a fallback to synthetic data generation when files are unavailable.
- **Machine Learning**: Implements a TensorFlow-based ANN for training and prediction, optimized for CPU performance on Windows environments.
- **User Interface**: Built with Streamlit, allowing users to input city names, PV system configurations, and environmental conditions (irradiance and temperature) to generate forecasts.
- **Deployment**: Configured for deployment on Streamlit Community Cloud, with automatic updates via GitHub integration.

The project was developed as part of a personal learning journey in machine learning and AI, focusing on renewable energy applications. It includes a virtual environment setup (`venv`) with dependencies managed via `requirements.txt`, ensuring reproducibility. The code is structured in `app.py` for the Streamlit interface and `solar_forecaster.py` for the ANN logic, with Git LFS support for large dataset files.

Check the live demo at [insert Streamlit URL once deployed] and explore the codebase to see implementation details, including caching fixes and data preprocessing. Contributions and feedback are welcome!

Technologies: Python, Streamlit, TensorFlow, Pandas, NumPy, Matplotlib, Geopy, Scikit-learn.
