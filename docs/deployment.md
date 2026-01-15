# Deployment Guide

This guide provides instructions for deploying and running the "Video Clip Selection and Sequencing" project on your Ubuntu server.

## 1. Environment Setup on Ubuntu Server

First, you need to set up the environment on your Ubuntu server. This includes installing Python, FFmpeg, and the required Python libraries.

### 1.1. Install System Dependencies

Connect to your Ubuntu server via SSH and run the following commands:

```bash
# Update package lists
sudo apt-get update

# Install Python and pip
sudo apt-get install -y python3 python3-pip

# Install FFmpeg
sudo apt-get install -y ffmpeg
```

### 1.2. Install Python Libraries

Create a virtual environment for the project (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Uploading the Project from Your Mac

You can use the `scp` (secure copy) command on your Mac to upload the project files to your Ubuntu server.

### 2.1. From Your Mac Terminal

Open a terminal on your Mac and navigate to the directory where your project is located. Then, run the following command:

```bash
scp -r /path/to/your/project/video-sequencer username@your_server_ip:/path/to/remote/directory
```

- Replace `/path/to/your/project/video-sequencer` with the actual path to your project directory on your Mac.
- Replace `username` with your username on the Ubuntu server.
- Replace `your_server_ip` with the IP address of your Ubuntu server.
- Replace `/path/to/remote/directory` with the directory on the server where you want to store the project (e.g., `/home/your_username/`).

### 2.2. Example

```bash
scp -r ~/Documents/video-sequencer giannis@192.168.1.100:/home/giannis/
```

This command will copy the `video-sequencer` directory and all its contents to the `/home/giannis/` directory on the server.

## 3. Running the Application

Once the project is uploaded, you can run the application from your Ubuntu server's terminal.

1.  **Connect to your server via SSH:**

    ```bash
    ssh username@your_server_ip
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd /path/to/remote/directory/video-sequencer
    ```

3.  **Activate the virtual environment (if you created one):**

    ```bash
    source venv/bin/activate
    ```

4.  **Run the main script:**

    ```bash
    python main.py --video-dir ./data/input/videos --audio ./data/input/audio/voiceover.mp3 --output ./data/output
    ```

    Make sure to replace the paths with the actual locations of your video directory, audio file, and desired output directory.

## 4. Uploading Test Video Clips

To upload your test video clips and voice-over audio, you can also use the `scp` command.

### 4.1. Create Directories on the Server

First, create the necessary directories on your server:

```bash
ssh username@your_server_ip "mkdir -p /path/to/remote/directory/video-sequencer/data/input/videos"
ssh username@your_server_ip "mkdir -p /path/to/remote/directory/video-sequencer/data/input/audio"
```

### 4.2. Upload Files

From your Mac terminal, run the following commands:

**Upload videos:**

```bash
scp /path/to/your/videos/* username@your_server_ip:/path/to/remote/directory/video-sequencer/data/input/videos/
```

**Upload audio:**

```bash
scp /path/to/your/audio/voiceover.mp3 username@your_server_ip:/path/to/remote/directory/video-sequencer/data/input/audio/
```

Now you are ready to run the application with your own test data.
