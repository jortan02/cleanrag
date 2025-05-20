# Streamlit Boilerplate Application

This is a boilerplate Streamlit application that provides a clean starting point for building data applications.

## Features

- Clean and modern UI layout
- Responsive sidebar with controls
- Two-column layout for content
- Example charts and data tables
- Easy to customize and extend
- Support for both local and hosted modes
- Environment-based configuration
- GPU support with CUDA version detection

## Getting Started

1. Make sure you have Python installed on your system
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Configure your environment variables in the `.env` file:
   - Set `LOCAL_MODE=True` for local development
   - Set `LOCAL_MODE=False` for hosted deployment
   - Set `USE_GPU=True` to enable GPU support (requires CUDA)
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

## GPU Requirements

If you plan to use GPU acceleration:

1. Ensure you have a CUDA-capable NVIDIA GPU
2. Install the appropriate CUDA toolkit for your system
3. Install the matching cuDNN version
4. Set `USE_GPU=True` in your `.env` file

The application will automatically detect and display:
- CUDA version
- GPU model
- Available GPU memory
- Detailed GPU information (in debug mode)

## Project Structure

- `app.py`: Main application file
- `config.py`: Configuration management
- `utils/gpu_utils.py`: GPU detection and information utilities
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
- `.env`: Environment variables (create from .env.example)
- `.env.example`: Example environment configuration

## Configuration

The application supports two modes:

### Local Mode
When `LOCAL_MODE=True`:
- Uses local models and resources
- Configure local model settings in `.env`:
  - `LOCAL_MODEL_PATH`: Path to local models
  - `USE_GPU`: Enable/disable GPU usage

### Hosted Mode
When `LOCAL_MODE=False`:
- Uses hosted API endpoints
- Configure API settings in `.env`:
  - `API_KEY`: Your API key
  - `API_ENDPOINT`: API endpoint URL

### Debug Mode
Set `DEBUG=True` in `.env` to enable debug information in the sidebar.

## Customization

You can customize this boilerplate by:
1. Modifying the UI elements in `app.py`
2. Adding new pages using Streamlit's multi-page app feature
3. Implementing your own data processing and visualization logic
4. Adding new configuration options in `config.py`

## Dependencies

The main dependencies are:
- streamlit
- pandas
- numpy
- python-dotenv
- torch (for GPU support)

All dependencies are listed in `requirements.txt`. 