import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import csv
import numpy as np
from PIL import Image
import torchaudio, torchvision
from torchaudio.prototype.pipelines import VGGISH

def generate_audio_set(dataPath, line, audioLength, vgg_sample_rate):

    data = line.split(',')
    videoName = data[0]
    dataName = data[0]
    audioFilePath = os.path.join(dataPath, videoName + '.wav')

    imageFrameTimeStamp = float(data[1])
    halfAudioLength = audioLength/2

    # samplingRate, audio = wavfile.read(audioFilePath)
    audio, samplingRate = torchaudio.load(audioFilePath)
    audio = audio.squeeze(0)
    if(len(audio.shape) > 1):
        audio = torch.mean(audio, dim=0)
    audioLengthSeconds = audio.size(0)/samplingRate
    
    shortage = 0
    excess = 0

    if (imageFrameTimeStamp - halfAudioLength) < 0:
        currentAudioStartTime = 0
        shortage = halfAudioLength - imageFrameTimeStamp
    else:
        currentAudioStartTime = imageFrameTimeStamp - halfAudioLength
    
    if (imageFrameTimeStamp + halfAudioLength) > audioLengthSeconds:
        currentAudioEndTime = audioLengthSeconds
        excess = imageFrameTimeStamp + halfAudioLength - audioLengthSeconds
    else:
        currentAudioEndTime = imageFrameTimeStamp + halfAudioLength

    audioStartIndex = int(samplingRate*currentAudioStartTime)
    audioEndIndex = int(samplingRate*currentAudioEndTime) - 1 #Subtracting 1 just to be safe with indices. 
    currentAudio = audio[audioStartIndex:audioEndIndex]
    shortageLength = int(samplingRate*shortage)
    excessLength = int(samplingRate*excess)
    currentAudio = torch.nn.functional.pad(currentAudio, (shortageLength, excessLength), mode='constant', value=0)
    currentAudio = torchaudio.functional.resample(currentAudio, samplingRate, vgg_sample_rate)

    return currentAudio

def load_audio(audio, audio_processor):
    # dataName = data[0]
    fps = 25
    #audio = audioSet[dataName]    
    processed_audio=audio_processor(audio)
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    # audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)

    # maxAudio = int(numFrames * 4)
    # if audio.shape[0] < maxAudio:
    #     shortage    = maxAudio - audio.shape[0]
    #     audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    # audio = audio[:int(round(numFrames * 4)),:]  
    return processed_audio

def load_visual(data, dataPath, numImages, visualAug, image_transform): 
    dataName = data[0]
    videoName = data[0]
    entityID = data[7]
    faceFolderPath = os.path.join(dataPath, videoName, entityID)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    numFaceFiles = len(sortedFaceFiles)
    currentFaceFilePath = os.path.join(faceFolderPath, "%.2f"%float(data[1])+".jpg")
    halfNumImages = numImages // 2

    shortage = 0
    excess = 0

    currentImageIndex = sortedFaceFiles.index(currentFaceFilePath)
    if numImages == 1:
        startIndex = currentImageIndex
        endIndex = startIndex + 1
    elif numImages > 1 : 
        if currentImageIndex - halfNumImages >= 0:
            startIndex = currentImageIndex - halfNumImages
        else:
            startIndex = 0
            shortage = halfNumImages - currentImageIndex
        
        if currentImageIndex + halfNumImages + 1 > numFaceFiles:
            endIndex = numFaceFiles
            excess = currentImageIndex + halfNumImages + 1 - numFaceFiles
        else:
            endIndex = currentImageIndex + halfNumImages + 1

    faces = []
    H = 112
    # if visualAug == True:
    #     new = int(H*random.uniform(0.7, 1))
    #     x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
    #     M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
    #     augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    # else:
    #     augType = 'orig'
    for faceFile in sortedFaceFiles[startIndex:endIndex]:
        # face = cv2.imread(faceFile)
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # face = cv2.resize(face, (H,H))
        # # face = (face - 128)/128
        # if augType == 'orig':
        #     faces.append(face)
        # elif augType == 'flip':
        #     faces.append(cv2.flip(face, 1))
        # elif augType == 'crop':
        #     faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        # elif augType == 'rotate':
        #     faces.append(cv2.warpAffine(face, M, (H,H)))
        face = Image.open(faceFile)
        # face = face.resize(size=(H,H))
        face = image_transform(face)
        faces.append(face)
    # faces = numpy.array(faces)
    # faces = faces.astype(np.float64)
    # faces = faces/255
    # faces = np.pad(faces, ((shortage, excess), (0,0), (0,0)))
    faces = torch.stack(faces, dim=0)
    return faces


def load_label(data):
    res = []
    label = int(data[8])
    res.append(label)
    res = numpy.array(res)
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, datasetPath, loadAudioSeconds, loadNumImages, **kwargs):
        trialFileName = trialFileName.replace('loader', 'labels')
        audioPath = audioPath.replace('clips', 'orig')
        self.audioPath  = audioPath
        self.visualPath = visualPath
        #self.miniBatch = []      
        
        self.mixLst = open(trialFileName).read().splitlines()
        self.mixLst = self.mixLst[1:]
        self.sampleAudioLength = loadAudioSeconds
        self.sampleNumImages = loadNumImages
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        # sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        # start = 0        
        # while True:
        #   length = int(sortedMixLst[start].split('\t')[1])
        #   end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
        #   self.miniBatch.append(sortedMixLst[start:end])
        #   if end == len(sortedMixLst):
        #       break
        #   start = end
        self.image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(112,112)), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(degrees=30), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.audioProcessor = VGGISH.get_input_processor()    
        self.vgg_sample_rate = VGGISH.sample_rate

    def __getitem__(self, index):
        # batchList    = self.miniBatch[index]
        # numFrames   = int(batchList[-1].split('\t')[1])
        #audioFeatures, visualFeatures, labels = [], [], []
        line = self.mixLst[index]
        #audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        audio = generate_audio_set(self.audioPath, line, self.sampleAudioLength, self.vgg_sample_rate) # load the audios in this batch to do augmentation
        #for line in batchList:
        data = line.split(',')            
        audioFeatures = load_audio(audio, self.audioProcessor)
        # print('dataloader audioFeatures shape: ', audioFeatures.shape)
        # print('dataloader audioFeatures: ', audioFeatures)  
        visualFeatures = load_visual(data, self.visualPath, self.sampleNumImages, visualAug = True, image_transform=self.image_transform)
        # print('dataloader visualFeatures shape: ', visualFeatures.shape)
        # print('dataloader visualFeatures: ', visualFeatures)
        labels = load_label(data)
        #print(numpy.array(audioFeatures).shape, numpy.array(visualFeatures).shape, numpy.array(labels).shape)
        return torch.FloatTensor(audioFeatures), \
               torch.FloatTensor(visualFeatures), \
               torch.LongTensor(labels)        

    def __len__(self):
        return len(self.mixLst)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, datasetPath, loadAudioSeconds, loadNumImages, **kwargs):
        trialFileName = trialFileName.replace('loader', 'labels')
        audioPath = audioPath.replace('clips', 'orig')
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.miniBatch = self.miniBatch[1:]
        self.sampleAudioLength = loadAudioSeconds
        self.sampleNumImages = loadNumImages

        self.dataPathAVA = datasetPath
        self.image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(112,112)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.audioProcessor = VGGISH.get_input_processor()
        self.vgg_sample_rate = VGGISH.sample_rate

    def __getitem__(self, index):
        line       = self.miniBatch[index]
        #numFrames  = int(line[0].split('\t')[1])
        audio   = generate_audio_set(self.audioPath, line, self.sampleAudioLength, vgg_sample_rate=self.vgg_sample_rate)        
        data = line.split(',')
        audioFeatures = load_audio(audio, self.audioProcessor)
        visualFeatures = load_visual(data, self.visualPath,self.sampleNumImages, visualAug = False, image_transform=self.image_transform)
        labels = load_label(data)         
        return torch.FloatTensor(audioFeatures), torch.FloatTensor(visualFeatures), torch.LongTensor(labels)

    def __len__(self):
        return len(self.miniBatch)