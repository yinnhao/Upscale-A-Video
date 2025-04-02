## AIGC videos
# python inference_upscale_a_video.py -i ./inputs/aigc_1.mp4 -o ./results -n 150 -g 6 -s 30 -p 24,26,28 --tile_size 128 --no_llava

# python inference_upscale_a_video.py \
# -i ./inputs/aigc_2.mp4 -o ./results -n 150 -g 6 -s 30 -p 24,26,28

# python inference_upscale_a_video.py \
# -i ./inputs/aigc_3.mp4 -o ./results -n 150 -g 6 -s 30 -p 20,22,24

## old videos/movies/animations 
python inference_upscale_a_video.py -i ./inputs/old_video_1.mp4 -o ./results -n 150 -g 9 -s 30 --no_llava

python inference_upscale_a_video.py -i ./inputs/old_movie_1.mp4 -o ./results -n 100 -g 5 -s 20 -p 17,18,19 --no_llava

# python inference_upscale_a_video.py -i ./inputs/old_movie_2.mp4 -o ./results -n 120 -g 6 -s 30 -p 8,10,12 --tile_size 128 --no_llava

python inference_upscale_a_video.py -i ./inputs/old_animation_1.mp4 -o ./results -n 120 -g 6 -s 20 --use_video_vae