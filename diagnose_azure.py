#!/usr/bin/env python3
"""
Azure OpenAI Tenant Diagnosis Utility

This script helps diagnose tenant mismatch issues with Azure OpenAI authentication.
It provides detailed information about your current Azure CLI login and suggests
solutions for common authentication problems.
"""

import json
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and return the result."""
    try:
        print(f"\n🔍 {description}...")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        return {"success": True, "output": result.stdout.strip(), "error": None}
    except subprocess.CalledProcessError as e:
        return {"success": False, "output": e.stdout.strip(), "error": e.stderr.strip()}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

def main():
    """Run Azure OpenAI tenant diagnostics."""
    print("🔧 Azure OpenAI Tenant Diagnosis Utility")
    print("=" * 50)
    
    # Check if Azure CLI is installed
    cli_check = run_command("az --version", "Checking Azure CLI installation")
    if not cli_check["success"]:
        print("❌ Azure CLI is not installed or not in PATH")
        print("💡 Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return 1
    
    print("✅ Azure CLI is installed")
    
    # Check current account
    account_check = run_command(
        "az account show --query \"{subscriptionId:id,tenantId:tenantId,name:name,user:user.name}\" --output json",
        "Checking current Azure CLI login"
    )
    
    if not account_check["success"]:
        print("❌ Not logged into Azure CLI")
        print("💡 Run: az login")
        return 1
    
    try:
        account_info = json.loads(account_check["output"])
        print("✅ Azure CLI is logged in")
        print(f"   Tenant ID: {account_info.get('tenantId', 'Unknown')}")
        print(f"   Subscription: {account_info.get('name', 'Unknown')}")
        print(f"   Subscription ID: {account_info.get('subscriptionId', 'Unknown')}")
        print(f"   User: {account_info.get('user', 'Unknown')}")
    except json.JSONDecodeError:
        print("❌ Could not parse Azure account information")
        return 1
    
    # List available tenants
    tenants_check = run_command(
        "az account list --query \"[].{Name:name,TenantId:tenantId,SubscriptionId:id}\" --output json",
        "Listing available tenants and subscriptions"
    )
    
    if tenants_check["success"]:
        try:
            tenants = json.loads(tenants_check["output"])
            print(f"\n📋 Available Tenants and Subscriptions:")
            for i, tenant in enumerate(tenants, 1):
                print(f"   {i}. {tenant.get('Name', 'Unknown')}")
                print(f"      Tenant ID: {tenant.get('TenantId', 'Unknown')}")
                print(f"      Subscription ID: {tenant.get('SubscriptionId', 'Unknown')}")
        except json.JSONDecodeError:
            print("❌ Could not parse tenant information")
    
    # Check .env configuration
    env_file = Path(".env")
    if env_file.exists():
        print(f"\n📄 Found .env file")
        try:
            with open(env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line.startswith("AZURE_OPENAI_ENDPOINT=") and not line.endswith("="):
                        endpoint = line.split("=", 1)[1]
                        print(f"   Endpoint: {endpoint}")
                        
                        # Check endpoint format
                        if ".cognitiveservices.azure.com" in endpoint:
                            print("   ⚠️  Warning: Endpoint uses .cognitiveservices.azure.com format")
                            print("   💡 Consider using .openai.azure.com format for Azure OpenAI")
                        elif ".openai.azure.com" in endpoint:
                            print("   ✅ Endpoint uses correct Azure OpenAI format")
                        else:
                            print("   ❌ Endpoint format may be incorrect")
                    
                    elif line.startswith("AZURE_OPENAI_DEPLOYMENT_NAME=") and not line.endswith("="):
                        deployment = line.split("=", 1)[1]
                        print(f"   Deployment: {deployment}")
        except Exception as e:
            print(f"   ❌ Error reading .env file: {e}")
    else:
        print(f"\n❌ .env file not found")
        print(f"💡 Copy .env.example to .env and configure your Azure OpenAI settings")
    
    # Provide specific guidance
    print(f"\n💡 Troubleshooting Steps:")
    print(f"1. Verify your Azure OpenAI resource tenant:")
    print(f"   - Go to Azure Portal: https://portal.azure.com")
    print(f"   - Find your Azure OpenAI resource")
    print(f"   - Note the tenant ID from the resource details")
    
    print(f"\n2. If tenant mismatch:")
    print(f"   - Login to correct tenant: az login --tenant YOUR_TENANT_ID")
    print(f"   - Or switch subscription: az account set --subscription YOUR_SUBSCRIPTION_ID")
    
    print(f"\n3. Verify endpoint format:")
    print(f"   - Correct: https://your-resource.openai.azure.com")
    print(f"   - Legacy: https://your-resource.cognitiveservices.azure.com (may work but not recommended)")
    
    print(f"\n4. Test configuration:")
    print(f"   - Run: python -m src.cli --test-azure")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())