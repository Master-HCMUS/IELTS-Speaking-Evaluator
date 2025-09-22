# IELTS Speaking Audio Recorder + AI Transcription

A professional, modular Python application for recording high-quality audio from your microphone and transcribing it using Azure OpenAI's Whisper model. Designed specifically for IELTS speaking practice and evaluation, featuring a clean modular architecture for maintainability and extensibility.

## 🎯 Features

### Core Features
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux
- **High-Quality Audio Recording**: Configurable sample rates, channels, and audio formats
- **AI-Powered Transcription**: Azure OpenAI Whisper integration for speech-to-text
- **Intuitive CLI Interface**: Easy-to-use command-line interface with clear prompts
- **Real-Time Feedback**: Visual indicators and recording status updates
- **Flexible Configuration**: Customizable audio and Azure settings
- **Graceful Error Handling**: Robust exception handling and recovery
- **Emergency Save**: Automatic saving when interrupted (Ctrl+C)

### Advanced Features
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Multiple Recording Modes**: Interactive mode, quick recording, and auto-transcription
- **File Management**: Organized storage and transcription management
- **Azure Authentication**: Managed identity and Azure CLI credential support
- **Comprehensive Logging**: Detailed transcription with timing and confidence scores
- **Storage Management**: File organization and cleanup utilities

## 🏗️ Architecture

The application follows a modular architecture with clear separation of concerns:

```
src/
├── cli.py                      # Main CLI orchestration layer
├── audio_recorder.py           # Core audio recording functionality
├── transcription_service.py    # Azure OpenAI Whisper integration
├── config_manager.py           # Configuration management
├── exceptions.py               # Custom exception hierarchy
├── ui/                         # User interface components
│   ├── __init__.py
│   ├── menu_system.py         # Menu display and user interactions
│   └── display_helpers.py     # Formatting and display utilities
└── workflows/                  # Business logic orchestration
    ├── __init__.py
    ├── workflow_orchestrator.py  # Main workflow coordination
    ├── configuration_handlers.py # Configuration workflows
    └── file_manager.py          # File operations and management
```

### Module Responsibilities

#### Core Modules
- **`cli.py`**: Thin orchestration layer that coordinates between UI and business logic
- **`audio_recorder.py`**: Hardware interface for microphone recording
- **`transcription_service.py`**: Azure OpenAI API integration and authentication
- **`config_manager.py`**: JSON-based configuration with validation
- **`exceptions.py`**: Custom exception hierarchy for error handling

#### UI Package (`src/ui/`)
- **`menu_system.py`**: Menu display, input collection, and navigation
- **`display_helpers.py`**: Formatting utilities for consistent output

#### Workflows Package (`src/workflows/`)
- **`workflow_orchestrator.py`**: Coordinates recording and transcription workflows
- **`configuration_handlers.py`**: Manages configuration user interactions
- **`file_manager.py`**: Handles file operations and organization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A working microphone
- Audio drivers properly installed
- Azure OpenAI resource (for transcription features)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ielts-speaking-recorder.git
   cd ielts-speaking-recorder
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Azure OpenAI (for transcription features):**
   ```bash
   # Copy the environment template
   cp .env.example .env
   
   # Edit the .env file with your Azure OpenAI credentials
   # nano .env  # or use your preferred editor
   ```
   
   Fill in your Azure OpenAI settings in the `.env` file:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT_NAME=whisper
   AZURE_OPENAI_API_VERSION=2024-02-01
   ```

4. **Set up Azure authentication:**
   ```bash
   az login
   ```

5. **Run the application:**
   ```bash
   python -m src.cli
   ```

## 📖 Usage

### Interactive Mode
```bash
python -m src.cli
```
Launches the full interactive menu with all features available.

### Quick Recording
```bash
python -m src.cli --quick
python -m src.cli --quick --output my_recording.wav
```

### Record and Transcribe
```bash
python -m src.cli --transcribe
```

### Configuration
```bash
python -m src.cli --config           # Audio settings
python -m src.cli --azure-config     # Azure OpenAI settings
python -m src.cli --devices          # List audio devices
python -m src.cli --test-azure       # Test Azure connection
```

## ⚙️ Configuration

### Audio Settings
The application supports configurable audio parameters:
- **Sample Rate**: 8000, 16000, 22050, 44100, 48000 Hz
- **Channels**: 1 (Mono) or 2 (Stereo)
- **Data Type**: int16, float32
- **Output Directory**: Custom recording directory

### Azure OpenAI Settings
For transcription features, configure:
- **Endpoint**: Your Azure OpenAI service endpoint
- **API Key**: Your Azure OpenAI API key
- **Deployment**: Whisper model deployment name
- **API Version**: Azure OpenAI API version
- **Language**: Target language or auto-detection
- **Auto-transcribe**: Enable automatic transcription after recording

### Configuration File
Settings are stored in `config/audio_config.json` and `.env`:
```json
{
  "audio": {
    "sample_rate": 44100,
    "channels": 1,
    "dtype": "int16"
  },
  "azure_openai": {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_key": "your-api-key-here",
    "deployment_name": "whisper",
    "api_version": "2024-06-01",
    "auto_transcribe": true,
    "language": "auto"
  }
}
```

## 🔐 Azure OpenAI Setup

### 1. Create Azure OpenAI Resource
1. Go to the [Azure Portal](https://portal.azure.com)
2. Create a new Azure OpenAI resource
3. Note your endpoint URL (e.g., `https://your-resource.openai.azure.com`)

### 2. Deploy Whisper Model
1. In your Azure OpenAI resource, go to "Model deployments"
2. Deploy a Whisper model (e.g., `whisper`)
3. Note your deployment name

### 3. Configure Environment Variables
1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your Azure OpenAI credentials:**
   ```env
   # Required settings
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_DEPLOYMENT_NAME=whisper
   AZURE_OPENAI_API_VERSION=2024-06-01
   
   # Optional settings
   AZURE_OPENAI_LANGUAGE=auto
   AZURE_OPENAI_AUTO_TRANSCRIBE=true
   ```

3. **Get your API key from Azure Portal:**
   - Go to your Azure OpenAI resource in the Azure Portal
   - Click on "Keys and Endpoint" in the left menu
   - Copy either "KEY 1" or "KEY 2"
   - Paste it into your `.env` file as the `AZURE_OPENAI_API_KEY` value

4. **Save the file and restart the application** to load the new settings.

### 4. Verify Configuration
Check your Azure OpenAI setup:
```bash
python -m src.cli --test-azure
```

## 📁 File Organization

### Default Structure
```
recordings/
├── recording_20240922_143022.wav
├── recording_20240922_143022_transcript.txt
├── my_speech.wav
└── my_speech_transcript.txt
```

### Transcription Files
Transcription files include:
- Original text transcription
- Language detection results
- Processing time and confidence scores
- Word-level and segment-level timing
- Metadata about the original audio file

## 🛠️ Development

### Architecture Benefits
The modular architecture provides:
- **Separation of Concerns**: Each module has a single responsibility
- **Testability**: Components can be tested in isolation
- **Maintainability**: Changes are localized to specific modules
- **Extensibility**: New features can be added without affecting existing code
- **Reusability**: Components can be reused in different contexts

### Adding New Features
1. **UI Components**: Add to `src/ui/` package
2. **Business Logic**: Add to `src/workflows/` package
3. **Core Services**: Add as new modules in `src/`
4. **Integration**: Update `cli.py` orchestration

### Error Handling
The application uses a comprehensive exception hierarchy:
- `AudioRecordingError`: Audio-related issues
- `AudioDeviceError`: Device availability problems
- `ConfigurationError`: Configuration validation issues
- `TranscriptionError`: Azure OpenAI API issues
- `AzureAuthenticationError`: Authentication failures

## 🧪 Testing

### Manual Testing
```bash
# Test audio functionality
python -m src.cli --devices
python -m src.cli --quick

# Test Azure integration
python -m src.cli --test-azure
python -m src.cli --transcribe
```

### Configuration Testing
```bash
# Test configurations
python -m src.cli --config
python -m src.cli --azure-config
```

## 📋 Requirements

### Core Dependencies
- `sounddevice>=0.4.6`: Cross-platform audio I/O
- `scipy>=1.10.0`: Scientific computing and WAV file support
- `numpy>=1.24.0`: Numerical computing

### Azure Dependencies
- `openai>=1.12.0`: Azure OpenAI API client
- `azure-identity>=1.15.0`: Azure authentication
- `azure-core>=1.29.0`: Azure SDK core functionality
- `python-dotenv>=1.0.0`: Environment variable management

## 🐛 Troubleshooting

### Audio Issues
- **No devices found**: Check microphone connections and drivers
- **Recording fails**: Try different sample rates or channels
- **Distorted audio**: Use recommended settings (44100 Hz, mono)

### Azure Issues
- **Authentication fails**: Check your API key in the `.env` file
- **API errors**: Verify endpoint URL, API key, and deployment name
- **Network issues**: Verify internet connection and firewall settings
- **Configuration not loaded**: Ensure `.env` file exists and contains valid settings
- **Environment variables not working**: Restart the application after editing `.env`
- **Invalid API key**: Get a fresh key from Azure Portal > Resource > Keys and Endpoint
- **Wrong endpoint format**: Use `https://YOUR-RESOURCE.openai.azure.com` (not cognitive services URL)

### Configuration Issues
- **Settings not saved**: Check directory permissions
- **Invalid configuration**: Use configuration menu to reset
- **Import errors**: Verify all dependencies are installed
- **.env file missing**: Copy `.env.example` to `.env` and configure your Azure settings
- **Environment variables ignored**: Check for typos in variable names and restart the app

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the modular architecture principles
4. Add appropriate error handling
5. Test your changes thoroughly
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

## 🙏 Acknowledgments

- Built with Python's `sounddevice` library for cross-platform audio
- Azure OpenAI for speech-to-text capabilities
- Inspired by IELTS speaking practice needs
- Designed with software engineering best practices