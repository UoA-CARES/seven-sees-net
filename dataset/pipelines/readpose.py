class ReadPose:
    def __init__(self):
        self.keypoints = ["nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
                ]

    def __call__(self, line):
        print(line)
        line = [l for l in line.replace(',',"").split(' ') if l != '' and l != '\n']
        print(line)
        imgpath = line[0]
        line = line[1:]
        line = [float(l) for l in line]


        #line [imgpath [0], [y,x, conf][1], ...[4], ... [52], headlb[54], headrt,lhandlb[58], lhandrt, rhandlb[62], rhandrt, bboxlb[66], bboxrt[68] ]

        
        posepoints = line[0:51]
        head = line[51:55]
        lhand =line[55:59]
        rhand = line[59:63]
        bodybbox = line[63:]
        print(posepoints)
        pose_values = dict()

        for i in range(0, 51, 3):
            if len(posepoints) == 0:
                print('No pose points found...')
                pose_values[self.keypoints[i//3]] = dict(y = [],
                                            x=[],
                                            confidence=[])
            else:
                pose_values[self.keypoints[i//3]] = dict(y = posepoints[i],
                                                x=posepoints[i+1],
                                                confidence=posepoints[i+2])
        
        return pose_values, head, lhand, rhand, bodybbox, imgpath