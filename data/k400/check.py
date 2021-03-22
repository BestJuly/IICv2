import os


train_split_path = list(open('train_kinetics.list','r'))
new_list = open('vcp_train.list','w')

for line in train_split_path:
    videoname = line.strip()
    vid = videoname.split(' ')[0][19:]
    class_idx = int(videoname.split(' ')[-1])

    rgb_folder = os.path.join('/raid/dataset/kinetics400/', vid)

     # filenames = ['frame000001.jpg']
    for parent, dirnames, filenames in os.walk(rgb_folder):
        if 'n_frames' in filenames:
            filenames.remove('n_frames')
        filenames = sorted(filenames)
    framenames = filenames
    length = len(framenames)
    if length < 64:
        print(vid, length)
    else:
        new_list.write(line)

new_list.close()
