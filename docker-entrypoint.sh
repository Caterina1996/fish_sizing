#!/bin/bash

WORKSPACE_DIR="/home/rosuser/fish_sizing/src/"

# Check if workspace needs to be initialized
if [ ! -d "$WORKSPACE_DIR/src" ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "  Initializing workspace (first run only)"
    echo "════════════════════════════════════════════════════════════════"
    
    mkdir -p "$WORKSPACE_DIR/src"
    cd "$WORKSPACE_DIR"
    
    # Initialize catkin workspace
    source /opt/ros/noetic/setup.bash
    echo "Initializing catkin workspace..."
    catkin init
    
    cd "$WORKSPACE_DIR/src"
    
    # Check if repositories are already mounted
    if [ ! -d "flir_camera_driver/.git" ]; then
        echo "ERROR: flir_camera_driver repository not found!"
        echo "Please ensure you have:"
        echo "  1. Cloned the repository locally"
        echo "  2. Mounted it as a volume in docker-compose.yml"
        exit 1
    fi
    
    # Inform about found repository
    echo "Found flir_camera_driver repository ✓"
    
    # Ensure we're on the correct branch for flir_spinnaker_camera
    cd flir_camera_driver
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "caterina" ]; then
        echo "Switching to caterina branch..."
        git checkout caterina
    fi
    cd ..

    # Build the workspace
    cd "$WORKSPACE_DIR"
    echo ""
    echo "Building workspace (this may take a few minutes)..."
    catkin build
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Workspace initialized successfully!"
    echo "  Location: $WORKSPACE_DIR"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
else
    # cd "$WORKSPACE_DIR"
    # catkin build
    echo "Workspace already exists at $WORKSPACE_DIR"
fi

# Source ROS environment
source /opt/ros/noetic/setup.bash

# Source the workspace if it exists and is built
if [ -f "$WORKSPACE_DIR/devel/setup.bash" ]; then
    source "$WORKSPACE_DIR/devel/setup.bash"
    echo "Workspace sourced and ready!"
else
    echo "WARNING: Workspace NOT built"
fi

# Execute the command
exec "$@"