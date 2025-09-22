"""
Configuration management for the audio recording system.

This module handles loading, saving, and validating configuration settings
for audio recording parameters, providing sensible defaults and user customization.
Azure OpenAI settings are loaded from environment variables for security.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from .exceptions import ConfigurationError


class ConfigManager:
    """
    Manages configuration settings for the audio recording system.
    
    Handles loading configuration from files, saving user preferences,
    and providing sensible defaults for all audio parameters.
    """
    
    DEFAULT_CONFIG = {
        "audio": {
            "sample_rate": 44100,
            "channels": 1,
            "dtype": "int16"
        },
        "output": {
            "directory": "recordings",
            "filename_template": "recording_{timestamp}.wav"
        },
        "ui": {
            "show_real_time_level": False,
            "auto_save": True
        },
        "azure_openai": {
            "endpoint": "",
            "api_key": "",
            "deployment_name": "whisper",
            "api_version": "2024-06-01",
            "auto_transcribe": True,
            "language": "auto"
        }
    }
    
    def __init__(self, config_file: str = "config/audio_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the configuration file.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Override Azure OpenAI settings with environment variables
        self._load_azure_env_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default configuration.
        
        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                config = self._merge_configs(self.DEFAULT_CONFIG.copy(), user_config)
                
                # Validate the configuration
                self._validate_config(config)
                
                return config
            else:
                # Create default configuration file
                self._save_config(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG.copy()
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load configuration file: {e}")
            print("Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default configuration.
        
        Args:
            default: Default configuration dictionary
            user: User configuration dictionary
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    default[key] = self._merge_configs(default[key], value)
                else:
                    default[key] = value
        return default
    
    def _load_azure_env_config(self) -> None:
        """
        Load Azure OpenAI configuration from environment variables.
        
        This method overrides any Azure OpenAI settings with values from
        environment variables, prioritizing security and ease of deployment.
        """
        azure_config = self.config.setdefault("azure_openai", {})
        
        # Load Azure OpenAI settings from environment variables
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        if endpoint:
            azure_config["endpoint"] = endpoint
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        if api_key:
            azure_config["api_key"] = api_key
        
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        if deployment_name:
            azure_config["deployment_name"] = deployment_name
        
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        if api_version:
            azure_config["api_version"] = api_version
        
        language = os.getenv("AZURE_OPENAI_LANGUAGE", "").strip()
        if language:
            azure_config["language"] = language
        
        # Handle boolean environment variable for auto_transcribe
        auto_transcribe = os.getenv("AZURE_OPENAI_AUTO_TRANSCRIBE", "").strip().lower()
        if auto_transcribe in ["true", "1", "yes", "on"]:
            azure_config["auto_transcribe"] = True
        elif auto_transcribe in ["false", "0", "no", "off"]:
            azure_config["auto_transcribe"] = False
        
        # Validate the updated configuration if Azure settings are present
        if azure_config.get("endpoint") and azure_config.get("deployment_name"):
            try:
                self._validate_config(self.config)
            except ConfigurationError as e:
                print(f"⚠️  Warning: Invalid Azure OpenAI configuration from environment: {e}")
                print("Please check your .env file settings.")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        audio_config = config.get("audio", {})
        
        # Validate sample rate
        sample_rate = audio_config.get("sample_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ConfigurationError(f"Invalid sample rate: {sample_rate}")
        
        # Validate channels
        channels = audio_config.get("channels")
        if not isinstance(channels, int) or channels not in [1, 2]:
            raise ConfigurationError(f"Invalid channels: {channels}. Must be 1 or 2.")
        
        # Validate dtype
        dtype = audio_config.get("dtype")
        if dtype not in ["int16", "float32"]:
            raise ConfigurationError(f"Invalid dtype: {dtype}. Must be 'int16' or 'float32'.")
        
        # Validate Azure OpenAI configuration
        azure_config = config.get("azure_openai", {})
        
        # Validate endpoint (can be empty for initial setup)
        endpoint = azure_config.get("endpoint", "")
        if endpoint and not isinstance(endpoint, str):
            raise ConfigurationError("Azure OpenAI endpoint must be a string")
        
        if endpoint and not (endpoint.startswith("https://") and endpoint.endswith(".azure.com")):
            raise ConfigurationError("Azure OpenAI endpoint must be a valid Azure URL (https://*.azure.com)")
        
        # Validate API key (can be empty for initial setup)
        api_key = azure_config.get("api_key", "")
        if api_key and not isinstance(api_key, str):
            raise ConfigurationError("Azure OpenAI API key must be a string")
        
        if api_key and len(api_key.strip()) < 10:
            raise ConfigurationError("Azure OpenAI API key appears to be too short or invalid")
        
        # Validate deployment name
        deployment_name = azure_config.get("deployment_name", "")
        if not isinstance(deployment_name, str) or not deployment_name.strip():
            raise ConfigurationError("Azure OpenAI deployment name cannot be empty")
        
        # Validate API version
        api_version = azure_config.get("api_version", "")
        if not isinstance(api_version, str) or not api_version.strip():
            raise ConfigurationError("Azure OpenAI API version cannot be empty")
        
        # Validate auto_transcribe
        auto_transcribe = azure_config.get("auto_transcribe", True)
        if not isinstance(auto_transcribe, bool):
            raise ConfigurationError("auto_transcribe must be a boolean value")
        
        # Validate language (can be "auto" for auto-detection)
        language = azure_config.get("language", "auto")
        if not isinstance(language, str):
            raise ConfigurationError("Language must be a string")
        
        # Valid language codes (including "auto" for auto-detection)
        valid_languages = {
            "auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "ca", "uk"
        }
        if language not in valid_languages:
            raise ConfigurationError(f"Invalid language code: {language}. Must be one of: {', '.join(sorted(valid_languages))}")
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get_audio_config(self) -> Dict[str, Any]:
        """
        Get audio configuration parameters.
        
        Returns:
            Dict[str, Any]: Audio configuration dictionary
        """
        return self.config["audio"].copy()
    
    def save_audio_config(self, audio_config: Dict[str, Any]) -> None:
        """
        Save audio configuration parameters.
        
        Args:
            audio_config: Audio configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate before saving
        temp_config = self.config.copy()
        temp_config["audio"] = audio_config
        self._validate_config(temp_config)
        
        # Update and save
        self.config["audio"] = audio_config
        self._save_config(self.config)
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration parameters.
        
        Returns:
            Dict[str, Any]: Output configuration dictionary
        """
        return self.config["output"].copy()
    
    def get_ui_config(self) -> Dict[str, Any]:
        """
        Get UI configuration parameters.
        
        Returns:
            Dict[str, Any]: UI configuration dictionary
        """
        return self.config["ui"].copy()
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section (e.g., 'audio', 'output', 'ui')
            key: Configuration key
            value: New value
            
        Raises:
            ConfigurationError: If section doesn't exist or validation fails
        """
        if section not in self.config:
            raise ConfigurationError(f"Unknown configuration section: {section}")
        
        # Update the value
        old_value = self.config[section].get(key)
        self.config[section][key] = value
        
        try:
            # Validate the updated configuration
            self._validate_config(self.config)
            
            # Save if validation passes
            self._save_config(self.config)
            
        except ConfigurationError:
            # Restore old value if validation fails
            if old_value is not None:
                self.config[section][key] = old_value
            else:
                del self.config[section][key]
            raise
    
    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        self._save_config(self.config)
    
    def get_azure_openai_config(self) -> Dict[str, Any]:
        """
        Get Azure OpenAI configuration parameters.
        
        Returns:
            Dict[str, Any]: Azure OpenAI configuration dictionary
        """
        return self.config["azure_openai"].copy()
    
    def save_azure_openai_config(self, azure_config: Dict[str, Any]) -> None:
        """
        Save Azure OpenAI configuration parameters.
        
        Args:
            azure_config: Azure OpenAI configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate before saving
        temp_config = self.config.copy()
        temp_config["azure_openai"] = azure_config
        self._validate_config(temp_config)
        
        # Update and save
        self.config["azure_openai"] = azure_config
        self._save_config(self.config)
    
    def update_azure_endpoint(self, endpoint: str) -> None:
        """
        Update Azure OpenAI endpoint.
        
        Args:
            endpoint: Azure OpenAI service endpoint URL
            
        Raises:
            ConfigurationError: If endpoint is invalid
        """
        self.update_config("azure_openai", "endpoint", endpoint)
    
    def update_azure_deployment(self, deployment_name: str) -> None:
        """
        Update Azure OpenAI deployment name.
        
        Args:
            deployment_name: Name of the Whisper deployment
            
        Raises:
            ConfigurationError: If deployment name is invalid
        """
        self.update_config("azure_openai", "deployment_name", deployment_name)
    
    def is_azure_configured(self) -> bool:
        """
        Check if Azure OpenAI is properly configured.
        
        Returns:
            bool: True if Azure OpenAI endpoint, API key, and deployment are configured
        """
        azure_config = self.config.get("azure_openai", {})
        endpoint = azure_config.get("endpoint", "").strip()
        api_key = azure_config.get("api_key", "").strip()
        deployment = azure_config.get("deployment_name", "").strip()
        
        return bool(endpoint and api_key and deployment)
    
    def get_transcription_settings(self) -> Dict[str, Any]:
        """
        Get settings for transcription operations.
        
        Returns:
            Dict containing transcription preferences
        """
        azure_config = self.config["azure_openai"]
        return {
            "auto_transcribe": azure_config.get("auto_transcribe", True),
            "language": azure_config.get("language", "auto"),
            "deployment_name": azure_config.get("deployment_name", "whisper")
        }
    
    def get_config_info(self) -> str:
        """
        Get a formatted string with current configuration information.
        
        Returns:
            str: Formatted configuration information
        """
        audio = self.config["audio"]
        output = self.config["output"]
        azure = self.config["azure_openai"]
        
        # Azure configuration status
        azure_status = "✅ Configured" if self.is_azure_configured() else "❌ Not configured"
        azure_endpoint = azure.get("endpoint", "Not set")
        if azure_endpoint and len(azure_endpoint) > 50:
            azure_endpoint = azure_endpoint[:47] + "..."
        
        info = f"""Configuration Information:
Audio Settings:
  - Sample Rate: {audio['sample_rate']} Hz
  - Channels: {audio['channels']} ({'Mono' if audio['channels'] == 1 else 'Stereo'})
  - Data Type: {audio['dtype']}

Output Settings:
  - Directory: {output['directory']}
  - Filename Template: {output['filename_template']}

Azure OpenAI Settings:
  - Status: {azure_status}
  - Endpoint: {azure_endpoint}
  - Deployment: {azure['deployment_name']}
  - API Version: {azure['api_version']}
  - Auto-transcribe: {'Yes' if azure['auto_transcribe'] else 'No'}
  - Language: {azure['language']}
  - Source: {'Environment Variables (.env)' if self._has_azure_env_vars() else 'Configuration File'}

Configuration File: {self.config_file}
"""
        return info
    
    def _has_azure_env_vars(self) -> bool:
        """
        Check if Azure OpenAI environment variables are set.
        
        Returns:
            bool: True if endpoint, API key, and deployment are set via environment
        """
        endpoint_env = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key_env = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment_env = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        return bool(endpoint_env and api_key_env and deployment_env)
    
    def get_azure_env_status(self) -> str:
        """
        Get detailed status of Azure environment variable configuration.
        
        Returns:
            str: Formatted status of Azure environment variables
        """
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", ""),
            "AZURE_OPENAI_LANGUAGE": os.getenv("AZURE_OPENAI_LANGUAGE", ""),
            "AZURE_OPENAI_AUTO_TRANSCRIBE": os.getenv("AZURE_OPENAI_AUTO_TRANSCRIBE", "")
        }
        
        status_lines = ["Azure OpenAI Environment Variables:"]
        
        for var_name, var_value in env_vars.items():
            if var_value:
                # Mask sensitive values for display
                if var_name == "AZURE_OPENAI_ENDPOINT" and len(var_value) > 50:
                    display_value = var_value[:47] + "..."
                elif var_name == "AZURE_OPENAI_API_KEY":
                    # Mask API key for security
                    display_value = var_value[:8] + "..." + var_value[-4:] if len(var_value) > 12 else "***MASKED***"
                else:
                    display_value = var_value
                status_lines.append(f"  ✅ {var_name}: {display_value}")
            else:
                status_lines.append(f"  ❌ {var_name}: Not set")
        
        status_lines.append("")
        status_lines.append("To configure Azure OpenAI:")
        status_lines.append("1. Copy .env.example to .env")
        status_lines.append("2. Get your credentials from Azure Portal:")
        status_lines.append("   - Go to your Azure OpenAI resource")
        status_lines.append("   - Click 'Keys and Endpoint'")
        status_lines.append("   - Copy the endpoint URL and Key 1")
        status_lines.append("3. Fill in your .env file with these credentials")
        status_lines.append("4. Restart the application")
        
        return "\n".join(status_lines)