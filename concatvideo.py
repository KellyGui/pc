from moviepy.editor import *
import glob
def concateVideo(path):
    videos = sorted(glob.glob(path+"/*.mp4"))
    clips = []
    for video in videos:
        tmp = VideoFileClip(video)
        clips.append(tmp)
    final_clip = concatenate_videoclips(clips)
    save_path = os.path.join(path,"all.mp4")
    final_clip.to_videofile(save_path,fps=clips[0].fps, remove_temp=False)
    return save_path
