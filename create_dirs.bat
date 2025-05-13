@echo off
echo Creating directory structure for CritterDetector...

mkdir models 2>nul
mkdir videos 2>nul
mkdir checkpoints 2>nul
mkdir highlights 2>nul
mkdir timelines 2>nul
mkdir evaluation_results 2>nul
mkdir annotations 2>nul
mkdir verification_frames 2>nul

echo Directory structure created successfully.
echo.
echo Next steps:
echo 1. Place your videos in the "videos" directory
echo 2. Run "check_cuda.bat" to verify CUDA setup
echo 3. Run "python -m owl_highlighter.run_highlighter --video_path videos/your_video.mp4 --verbose"
echo.

pause 