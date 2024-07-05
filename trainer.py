import os, glob, time
import argparse
from model import *
from dataLoader_Image_audio import train_loader, val_loader

def parser():

    args = argparse.ArgumentParser(description="ASD Trainer")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    args.add_argument('--maxEpoch', type=int, default=10, help='Maximum number of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=128, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/notebooks/AVDIAR_ASD/", help='Path to the ASD Dataset')
    args.add_argument('--loadAudioSeconds', type=float, default=3, help='Number of seconds of audio to load for each training sample')
    args.add_argument('--loadNumImages', type=int, default=1, help='Number of images to load for each training sample')
    args.add_argument('--savePath', type=str, default="exps/exp1")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")
    args.add_argument('--pretrained_model_path', type=str, default="path not specified", help="Path to the pretrained parameters. These parameters are used to initialize training.")
    args.add_argument('--num_blocks_unfrozen', type=int, default=0, help='The number of convolution blocks unfrozen in the feature extractors for finetuning. Max value is 4 since the number of Conv Blocks in VGGish is 4.')

    args = args.parse_args()

    return args

def main(args):

    loader = train_loader(trialFileName = os.path.join(args.datasetPath, 'csv/train_loader.csv'), \
                          audioPath      = os.path.join(args.datasetPath , 'clips_audios/'), \
                          visualPath     = os.path.join(args.datasetPath, 'clips_videos/train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = args.batchSize, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = os.path.join(args.datasetPath, 'csv/val_loader.csv'), \
                        audioPath     = os.path.join(args.datasetPath , 'clips_audios'), \
                        visualPath    = os.path.join(args.datasetPath, 'clips_videos', args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = args.batchSize, shuffle = False, num_workers = 4)
    
    if args.evaluation == True:
        s = model(**vars(args))

        if args.eval_model_path=="path not specified":
            print('Evaluation model parameters path has not been specified')
            quit()
        
        s.loadParameters(args.eval_model_path)
        print("Parameters loaded from path ", args.eval_model_path)
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()    
    
    # Either loads a previous checkpoint or starts training from scratch
    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    # modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    # modelfiles.sort()  
    
    epoch = 1
    s = model(epoch = epoch, **vars(args))
    s.loadParameters(args.pretrained_model_path, map_location=s.device)

    for name, param in s.named_parameters():
        print('Final model {}: {}'.format(name, param.requires_grad))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")
    bestmAP = 0
    while(1):        
        loss, lr, train_acc = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            if mAPs[-1] > bestmAP:
                bestmAP = mAPs[-1]
                s.saveParameters(args.modelSavePath + "/best.model")
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%, trainAcc %2.2f\n"%(epoch, lr, loss, mAPs[-1], max(mAPs), train_acc))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__=="__main__":

    args = parser()

    main(args)