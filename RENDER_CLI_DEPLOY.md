# Deploying Your AI Image Upscaler API with Render CLI

This guide walks you through using the Render Command Line Interface (CLI) to deploy your AI Image Upscaler API.

## Prerequisites

- All the files from `render_deployment.zip` extracted to your local machine
- Basic familiarity with command line interfaces

## Step 1: Install the Render CLI

Open your terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/render-oss/cli/refs/heads/main/bin/install.sh | sh
```

After installation, open a new terminal tab and verify by running:

```bash
render
```

## Step 2: Authenticate with Render

Log in to your Render account through the CLI:

```bash
render login
```

This will open your browser where you'll need to:
1. Click "Generate token"
2. Wait for the "Success" message
3. Return to your terminal
4. Select your workspace when prompted

## Step 3: Prepare Your Deployment Files

Ensure you have all necessary files in a single directory:
- `render_app.py`
- `requirements.txt`
- `render.yaml`
- `.env` (create this from `.env.sample` with your actual RapidAPI key)

## Step 4: Create a New Service

There are two ways to deploy using the CLI:

### Option A: Using render.yaml (Blueprint)

1. Navigate to your project directory
2. Create a new blueprint deployment:

```bash
render blueprint create
```

3. Follow the prompts to select your repository or local files

### Option B: Manual Service Creation

```bash
render services create
```

When prompted:
- Select "Web Service"
- Choose "Upload Files" option
- Select your project directory
- Configure:
  - Name: ai-image-upscaler-api
  - Runtime: Python
  - Build Command: pip install -r requirements.txt
  - Start Command: gunicorn -k uvicorn.workers.UvicornWorker render_app:app -b 0.0.0.0:$PORT

## Step 5: Set Environment Variables

Set your environment variables:

```bash
render env set RAPIDAPI_KEY=your_rapidapi_key_here --service your-service-id
render env set RENDER_ENV=preview --service your-service-id
```

To get your service ID, run:
```bash
render services
```

## Step 6: Deploy Your Service

Trigger a deploy:

```bash
render deploys create your-service-id --wait
```

The `--wait` flag will make the command wait until deployment is complete.

## Step 7: Monitor Your Deployment

To check the status and logs of your deployment:

```bash
render logs your-service-id
```

## Step 8: Test Your API

Once deployed, test your API using the service URL:

```bash
curl https://your-service-name.onrender.com/
```

You should see a response indicating the API is running.

## Step 9: Connect to RapidAPI

Follow the instructions in `DEPLOYMENT_SUMMARY.md` to connect your deployed API to RapidAPI.

## Automating Deployments

For CI/CD pipelines or scripted deployments, use the non-interactive mode:

```bash
export RENDER_API_KEY=your_api_key_here
render deploys create your-service-id --wait --output json --confirm
```

## Updating Your Deployment

To update your deployment after making changes:

```bash
render deploys create your-service-id --wait
```

## Useful Commands

- List all services: `render services`
- View deployment logs: `render logs your-service-id`
- SSH into your service: `render ssh your-service-id`
- Delete a service: `render services delete your-service-id`

## Troubleshooting

- **Authentication issues**: Run `render login` again to refresh your token
- **Deployment failures**: Check logs with `render logs your-service-id`
- **Command errors**: Use `render help [command]` for detailed usage information 