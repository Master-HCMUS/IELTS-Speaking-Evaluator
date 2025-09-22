"""
Configuration handlers for the IELTS Speaking Audio Recorder.

This module handles all configuration-related user interactions
and settings management.
"""

from pathlib import Path
from typing import Dict, Any

from ..config_manager import ConfigManager
from ..ui.menu_system import MenuSystem
from ..exceptions import ConfigurationError


class ConfigurationHandlers:
    """
    Handles configuration-related workflows and user interactions.
    
    This class provides methods for configuring audio settings,
    Azure OpenAI settings, and other application preferences.
    """
    
    def __init__(self, config_manager: ConfigManager, menu_system: MenuSystem):
        """
        Initialize the configuration handlers.
        
        Args:
            config_manager: Configuration manager instance
            menu_system: Menu system for user interactions
        """
        self.config_manager = config_manager
        self.menu_system = menu_system
    
    def configure_audio_settings(self) -> None:
        """Interactive configuration of audio settings."""
        self.menu_system.display_section_header("Audio Configuration")
        
        current_config = self.config_manager.get_audio_config()
        
        try:
            # Sample rate configuration
            current_rate = current_config['sample_rate']
            self.menu_system.display_config_option("Current sample rate", f"{current_rate} Hz")
            
            valid_rates = [8000, 16000, 22050, 44100, 48000]
            print(f"Valid options: {', '.join(map(str, valid_rates))}")
            
            sample_rate_input = self.menu_system.get_user_input(
                "Enter new sample rate or press Enter to keep current: "
            )
            
            if sample_rate_input:
                sample_rate = int(sample_rate_input)
                if sample_rate not in valid_rates:
                    self.menu_system.display_warning("Unusual sample rate. Using anyway.")
                current_config['sample_rate'] = sample_rate
            
            # Channels configuration
            current_channels = current_config['channels']
            channel_text = 'Mono' if current_channels == 1 else 'Stereo'
            self.menu_system.display_config_option("Current channels", f"{current_channels} ({channel_text})")
            
            channels_input = self.menu_system.get_user_input(
                "Enter number of channels (1 for mono, 2 for stereo) or press Enter to keep current: "
            )
            
            if channels_input:
                channels = int(channels_input)
                if channels not in [1, 2]:
                    self.menu_system.display_warning("Only 1 (mono) or 2 (stereo) channels are typically supported.")
                current_config['channels'] = channels
            
            # Data type configuration
            current_dtype = current_config['dtype']
            self.menu_system.display_config_option("Current data type", current_dtype)
            
            dtype_input = self.menu_system.get_user_input(
                "Enter data type (int16, float32) or press Enter to keep current: "
            )
            
            if dtype_input and dtype_input in ['int16', 'float32']:
                current_config['dtype'] = dtype_input
            elif dtype_input:
                self.menu_system.display_warning("Invalid data type. Keeping current setting.")
            
            # Save configuration
            self.config_manager.save_audio_config(current_config)
            self.menu_system.display_success("Configuration saved successfully!")
            
        except ValueError as e:
            self.menu_system.display_error(f"Invalid input: {e}")
        except Exception as e:
            self.menu_system.display_error(f"Configuration error: {e}")
        
        self.menu_system.wait_for_enter()
    
    def configure_azure_openai(self) -> None:
        """Guide users to configure Azure OpenAI via .env file."""
        self.menu_system.display_section_header("Azure OpenAI Configuration")
        
        # Show current status from environment variables
        print(self.config_manager.get_azure_env_status())
        print()
        
        # Check if already configured via environment
        if self.config_manager.is_azure_configured():
            self.menu_system.display_success("Azure OpenAI is configured via environment variables!")
            
            if self.menu_system.get_yes_no_choice("Test Azure OpenAI connection now?"):
                from .workflow_orchestrator import WorkflowOrchestrator
                orchestrator = WorkflowOrchestrator(self.config_manager, self.menu_system)
                orchestrator.test_azure_connection()
        else:
            self.menu_system.display_warning("Azure OpenAI is not configured.")
            print()
            self.menu_system.display_info("To configure Azure OpenAI:")
            self.menu_system.display_info("1. Copy '.env.example' to '.env' in the project root")
            self.menu_system.display_info("2. Edit the .env file with your Azure OpenAI credentials:")
            self.menu_system.display_info("   - AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com")
            self.menu_system.display_info("   - AZURE_OPENAI_DEPLOYMENT_NAME=whisper")
            self.menu_system.display_info("3. Restart the application to load the new settings")
            print()
            self.menu_system.display_info("Authentication uses Azure CLI. Make sure you're logged in with: az login")
            
            if self.menu_system.get_yes_no_choice("Open .env.example file location?", False):
                import os
                import subprocess
                import platform
                
                env_example_path = Path(".env.example").resolve()
                if env_example_path.exists():
                    try:
                        if platform.system() == "Windows":
                            subprocess.run(["explorer", "/select,", str(env_example_path)], check=True)
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", "-R", str(env_example_path)], check=True)
                        else:  # Linux
                            subprocess.run(["xdg-open", str(env_example_path.parent)], check=True)
                        self.menu_system.display_success("File location opened!")
                    except Exception as e:
                        self.menu_system.display_error(f"Could not open file location: {e}")
                        self.menu_system.display_info(f"Manual path: {env_example_path}")
                else:
                    self.menu_system.display_error(".env.example file not found in project root")
        
        self.menu_system.wait_for_enter()
    
    def view_current_settings(self) -> None:
        """Display current audio and Azure settings."""
        self.menu_system.display_section_header("Current Settings")
        
        print(self.config_manager.get_config_info())
        
        # Show recorder status if available
        try:
            from .workflow_orchestrator import WorkflowOrchestrator
            orchestrator = WorkflowOrchestrator(self.config_manager, self.menu_system)
            if orchestrator.recorder:
                info = orchestrator.recorder.get_recording_info()
                status = '🔴 Active' if info['is_recording'] else '⚪ Inactive'
                print(f"Recording Status: {status}")
        except:
            pass  # Ignore if recorder is not initialized
        
        self.menu_system.wait_for_enter()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        if self.menu_system.get_yes_no_choice("Reset all settings to defaults? This cannot be undone.", False):
            try:
                self.config_manager.reset_to_defaults()
                self.menu_system.display_success("Settings reset to defaults!")
            except Exception as e:
                self.menu_system.display_error(f"Error resetting settings: {e}")
        else:
            self.menu_system.display_info("Reset cancelled.")
        
        self.menu_system.wait_for_enter()