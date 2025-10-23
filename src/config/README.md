# Configuration Security

## Important: Protecting API Keys

This directory contains configuration files that may include sensitive information such as:

- Azure OpenAI API keys
- Azure Speech Service API keys
- Other authentication credentials

## Setup Instructions

1. Copy `audio_config.json.example` to `audio_config.json`
2. Replace placeholder values with your actual API keys:
   - `YOUR_AZURE_OPENAI_API_KEY_HERE` with your Azure OpenAI API key
   - `YOUR_AZURE_SPEECH_API_KEY_HERE` with your Azure Speech API key
   - `YOUR_AZURE_REGION` with your Azure region (e.g., "eastus")
   - Update endpoint URLs to match your resources

3. The `audio_config.json` file is automatically ignored by git to prevent accidentally committing credentials

## Security Best Practices

- Never commit files containing real API keys
- Use environment variables when possible
- Regularly rotate your API keys
- Keep example files updated but without real credentials
- Review `.gitignore` rules regularly

## Files in this directory

- `audio_config.json.example` - Template configuration file (safe to commit)
- `audio_config.json` - Your actual configuration (ignored by git)