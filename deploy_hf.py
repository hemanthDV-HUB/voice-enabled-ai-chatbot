import argparse
import os
import sys
from huggingface_hub import HfApi, login

def deploy_to_huggingface(token: str, repo_id: str):
    """
    Automates uploading and deploying the Voice-Enabled Chatbot to Hugging Face Spaces.
    """
    print(f"\n🚀 Initiating deployment to Hugging Face Spaces: {repo_id}")
    
    # Authenticate
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Authenticated as Hugging Face user: {user_info.get('name', user_info.get('fullname', 'User'))}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please ensure your Hugging Face write token is valid.")
        sys.exit(1)

    # Create space repository if it does not already exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False
        )
        print(f"✅ Space repository verified/created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"⚠️ Note during repo verification: {e}")

    # Prepare list of project files to upload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_upload = [
        "app.py",
        "model.py",
        "requirements.txt",
        "README.md",
        os.path.join("data", "intents.json"),
        os.path.join("model", "chatbot_metadata.json"),
        os.path.join("model", "chatbot_model.pth"),
        os.path.join("model", "model_weights.json")
    ]

    print("\n📦 Uploading project files to Hugging Face Space...")
    for rel_path in files_to_upload:
        full_path = os.path.join(current_dir, rel_path)
        if os.path.exists(full_path):
            print(f" -> Uploading {rel_path}...")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=rel_path.replace("\\", "/"),
                repo_id=repo_id,
                repo_type="space"
            )
        else:
            print(f" ⚠️ Warning: {rel_path} not found locally.")

    public_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Public Live URL: {public_url}")
    print("Hugging Face is building your container. It will be live and ready in ~1 minute.")
    return public_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Voice Chatbot to Hugging Face Spaces")
    parser.add_argument("--token", type=str, help="Hugging Face Access Token (Write permissions)", required=False)
    parser.add_argument("--repo", type=str, help="Hugging Face repo id, e.g. username/voice-chatbot-slp", required=False)
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    repo = args.repo or os.environ.get("HF_REPO")

    if not token:
        print("\n🔑 Hugging Face Token required.")
        token = input("Enter your Hugging Face write token (from https://huggingface.co/settings/tokens): ").strip()

    if not repo:
        repo = input("Enter your target Space repo id (e.g. your_username/voice-chatbot-slp): ").strip()

    if token and repo:
        deploy_to_huggingface(token, repo)
    else:
        print("Missing token or repository id. Aborting deployment.")
