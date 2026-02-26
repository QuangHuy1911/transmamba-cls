Engagement Intensity Prediction with
Facial Behavior Features
Van Thong Huynh
School of Electronics and Computer Engineering
Chonnam National University
Gwangju, Korea
hvthong298@outlook.com.vn
Guee-Sang Lee
School of Electronics and Computer Engineering
Chonnam National University
Gwangju, Korea
gslee@jnu.ac.kr
ABSTRACT
This paper describes an approach for the engagement pre
diction task, a sub-challenge of the 7th Emotion Recognition
in the Wild Challenge (EmotiW 2019). Our method involves
three fundamental steps: feature extraction, regression and
model ensemble. In the first step, an input video is divided
into multiple overlapped segments (instances) and the fea
tures extracted for each instance. The combinations of Long
short-term memory (LSTM) and Fully connected layers de
ployed to capture the temporal information and regress the
engagement intensity for the features in previous step. In
the last step, we performed fusions to achieve better perfor
mance. Finally, our approach achieved a mean square error
of 0.0597, which is 4.63% lower than the best results last year.
CCSCONCEPTS
• Computing methodologies → Activity recognition
and understanding.
KEYWORDS
Engagement detection, E-learning Environment, Ensemble
model, Facial behavior, Affective computing
∗Corresponding author
Permission to make digital or hard copies of all or part of this work for
personal or classroom useisgrantedwithoutfeeprovidedthatcopiesarenot
made or distributed for profit or commercial advantage and that copies bear
this notice and the full citation on the first page. Copyrights for components
of this work owned by others than ACM must be honored. Abstracting with
credit is permitted. To copy otherwise, or republish, to post on servers or to
redistribute to lists, requires prior specific permission and/or a fee. Request
permissions from permissions@acm.org.
ICMI ’19, October 14–18, 2019, Suzhou, China
©2019 Association for Computing Machinery.
ACMISBN978-1-4503-6860-5/19/05.
https://doi.org/10.1145/3340555.3355714
Hyung-Jeong Yang
School of Electronics and Computer Engineering
Chonnam National University
Gwangju, Korea
hjyang@jnu.ac.kr
Soo-Hyung Kim∗
School of Electronics and Computer Engineering
Chonnam National University
Gwangju, Korea
shkim@jnu.ac.kr
ACMReference Format:
Van Thong Huynh, Hyung-Jeong Yang, Guee-Sang Lee, and Soo
Hyung Kim. 2019. Engagement Intensity Prediction with Facial Be
havior Features. In 2019 International Conference on Multimodal In
teraction (ICMI ’19), October 14–18, 2019, Suzhou, China. ACM, New
York, NY, USA, 5 pages. https://doi.org/10.1145/3340555.3355714
1 INTRODUCTION
Automatic user engagement prediction is an active research
area that aims to predict the engagement of the connection
between two or more instances. It attracted the attention of
many researchers because of its applications in many fields
such as education, human-computer interaction, social inter
action, as well as health care. In the learning environment,
engagement means student engagement which organized
into behavioral- enthusiasm for participating in the learning
process and school-related activities; emotional involves the
affective states in the learning task or learning context includ
ing interest and boredom; cognitive refers to the investment
in learning [8]. The traditional classroom has fundamentally
transformed into the e-learning environment such as Mas
sive Online Open Courses (MOOC), which takes advantage
of new technologies to transfer the knowledge efficiently. In
this situation, we need to explore newapproachestomeasure
the engagementlevel duringlearning time towardimproving
the course quality.
The effects of feature types (Box Filter, Energy Filters fea
tures) and classifier combinations from the automatic facial
expression to predict the student engagement studied in [15].
The dataset with four levels of engagement collected while
subjects were playing cognitive skills training software on an
iPad. The author in [4] analyzed 19 Action Units (AUs), head
pose, and position as well as the body movement. Dataset
with five affective state levels collected in a school computer
lab while students were interacting with educational game.
567
ICMI ’19, October 14–18, 2019, Suzhou, China
EmotiW Grand Challenge
In [12], the facial features and heart rate employed to de
tect the engagement while students completed a structured
writing activity. They analyzed the Animation Units from
(ANUs) Kinect Face tracker, Local Binary Patterns in Three
Orthogonal Planes (LBP-TOP), and heart rate features. The
overall best results were found using a fusion of the best two
channels (LBP-TOP, heart rate).
An “in the wild” dataset with four engagement intensities
was collected [11]. They fed the fused features of Head Pose
and Eye Gaze to a deep multi-instance network architecture
to regress the intensity value. [13] proposed a set of Eye
Gaze, AUs, Head Pose features with Gated Recurrent Unit
networks, while [14] fed a set of AU intensities features to
a Temporal Convolution Networks. In another way, [17]
performed a late fusion of Eye Gaze, Head Pose with Body
Posture. In [6], authors made the ensembles of the traditional
machinelearningmethod(K-mean,AdaBoost),deeplearning
method (Bidirectional LSTM) and rules-based method.
In this paper, we focus on analyzing facial behavior fea
tures for engagement prediction with the experiments on
the dataset provided in [11]. Our paper organized as follows:
Section 2 describes the feature extraction and the network
architecture, then experiments and results show in Section
3, Section 4 summarizes conclusions and the future works.
2 PROPOSEDMETHOD
In our approach, each video goes through OpenFace [3], a
well-known toolkit for facial behavior analysis in the com
puter vision community, which implemented MTCNN [19]
for face detection, [2, 18] for facial landmark detection and
tracking, [16] for eye gaze estimation, and [1] for facial ac
tion unit detection. Based on the face region extracted by
OpenFace, the facial feature obtained by a pre-trained SE
ResNet-50 model [5, 9, 10]. We divided the video sequencev
into ℓ segmentss1,s2, . . .,sℓ with |si∩si+1| = r and |si = si+1|,
i = 1,ℓ −1. The features for each segment obtained based
on the statistical characteristics of OpenFace features and
SE-ResNet feature on frames belong to them. 1 shows our
overall system pipeline.
Feature extraction
Previous works in engagement prediction showed that the
engagement had a highly correlated with the movement in
the face (eye gaze, head pose) [6, 11, 13, 17] as well as the
facial expression (LBP-TOP, Gabor features) [11, 12, 15]. In
this paper, we consider the combination of eye gaze, head
pose features with the automatic feature extracted by convo
lutional neural networks.
Eye gaze and head pose features. OpenFace provides 56 land
marks to determine eye regions in each frame (2D landmarks
Input	Video
Feature	extraction
OpenFace
Statistical	feature
process
	feature	set
1
Model	
ℓ1
1
×60
Model	
Regression	Model
Face	Regions
SE-ResNet-50-128D
		Facial	Features
×128
	feature	set
2
ℓ2
Model	
2
Ensemble	models
3
×128
Model	
4
Engagement	intensity
Figure 1: The overall of our system pipeline.
in pixels) as well as in the real world (3D landmarks in mil
limeters). The eye gaze vector for each eye as well as gaze
angle in radian for both eyes were estimated. We deployed
this information to get the F1 feature set.
Tocapturethechangingofheadposeandgazedirection,in
eachsegment,wecalculatedthemeanandstandarddeviation
of head location, rotation, and gaze vector of both eyes as
well as gaze angle. In additional, we observed that the eye
shape- the y distance between two eye landmark points
whose the x distance is minimum, and eye locations in the
frame as well as in the real world, are one of the reasons that
makes the change in the gaze direction. These properties
were observed based on the mean, standard deviation, and
the ratio between the minimumandmaximumvaluesineach
segment. 1 shows the dimension of each feature type in F1.
Facial features. The feature set F2 was obtained by feeding
the faceregionsintoaCNNmodel.Thesefaceswereachieved
from68landmarks,providingbyOpenFace.Beforeinputting
to the network, the face images were resized so that the
shorter size was 256 pixels and then were cropped out the
center 224 × 224 portion of the image.
568
EIPwithFacialBehaviorFeatures ICMI’19,October14–18,2019,Suzhou,China
Table1:AsummaryofF1 featuresetextractedbasedon
OpenFace.
Featuretype Featureinformation Dim
Gazedirection mean,standarddeviation 16
Eye landmarks 2D
and 3D, distance
fromeyetocamera
mean, standard deviation,
coefficientofvariation, ra
tiobetweentheminandmax
values
32
Headpose mean,standarddeviation 12
ThefacialfeatureswereobtainedbySE-ResNet-50[5,9,
10]whichintegratedtheSqueeze-and-Excitation(SE)blocks,
thatadaptivelyrecalibratedchannel-wisefeatureresponses
byexplicitlymodelingtherelationshipbetweenthechannels
ofitsconvolutionalfeature, intoResNet-50.Theefficiency
ofSEblocksdemonstratedforobjectandsceneclassification
bywinningintheILSVRC2017classificationcompetition.
TotakeadvantageofSEblocksandexistingstate-of-the
artdeeparchitectures,weusedSE-ResNet-50whichtrained
onMS-Celeb-1M[9], alargescalereal-worldfaceimage
dataset,andthenfine-tuneonVGGFace2dataset,alarge
scalefacerecognitiondatasetwithalargevariationinpose,
age, illumination,ethnicity,andprofession.Theauthorsin
[5]performedthetrainingandfine-tune.
Engagementintensityregressionmodels
Forthemodelarchitecture,wedeployedtwodifferentnet
worksA1,A2 foreachfeatureset.Onemodel isLSTM-FC,
learningintheoriginal featureset,whiletheotherisex
tendedbyusingthefullyconnectedrightbeforeLSTM(FC
LSTM-FC),learninginthenewrepresentationofthedata,as
shownin2.WeusedReLUactivationineachFClayerand
tanhfunctioninLSTMlayers.Inthevideo,theengagement
variedacrosstimedependingonthecurrentlearningcontent
aswellastheprevious.Tocapturethesevariations,weused
the2LSTMlayersfollowedby2FClayerstolearn,obtainthe
relationshipsbetweenconsecutivesegments,andmakethe
predictionforeachsegmentproducedbythelastFClayer.
Thesevalueswerepassedtoapoolinglayerwhichtookthe
averageofthemandoutputtedtheengagementintensityfor
thewholevideo.Inthiswork,wetookamulti-levelensem
blebasedontheresultsfromeverysinglemodeltoimprove
predictions.Theseimprovementswereduetothediversity
inthefeaturetypeaswellasfeaturerepresentation.
3 EXPERIMENTRESULTS
Dataset
Inthispaper,weexperimentedwithan“Engagement in
thewilddataset”[11]whichusedinEngagementprediction
1
2
Feature	set
FC
LSTM
LSTM
FC
FC
Global	average	pooling
LSTM
LSTM
FC
FC
Global	average	pooling
Engagement	Intensity Engagement	Intensity
Figure2:ThepipelineofA1,A2.
task, asub-challenge inEmotiW2019[7].Thedatawas
recordedwithawebcamonalaptoporcomputer,amobile
phonecamerawhilethestudentparticipantswerewatching
fiveminuteslongMOOCvideo.Theenvironmentinwhich
studentswatchedlearningmaterialalsodifferentfromthe
computer lab, canteen,playgroundtohostel rooms.The
datasetincluded91subjects(27femalesand64males)with
147,48and67videos,eachapproximately5minuteslong,for
training,validation,andtesting,respectively.Fourlevelsof
engagement{0,1,2,3}werelabeledbyannotators.3shows
thedistributionofthedevelopmentdata. Inthistask, the
problemwasformulatedasanregressionproblemwiththe
output in[0,1]correspondingto4engagement levels(0 :
0,0.33:1,0.66:2,1:3).Thesystemperformanceevaluated
withmeansquareerror(MSE)betweenthegroundtruth,
andthepredictedvalueofthetestset.
Experimentresults
Intheexperiments,weempiricallyselectedℓ1 =15and
ℓ2=21whicharethenumberofsegmentsforF1andF2,
respectively.Wehavetwodifferentmodelsforeachfeature
set,asshownin1.Ourmodeltrainedonthetrainingsetand
adjustedparameterswiththevalidationsetwithoutanydata
augmentationaswellasresamplingonthetrainingset.2
showsthedetailofeachmodel.TheresultsillustratethatA1
achievedbetterMSEthanA2forbothfeaturesetF1,F2.
Toimprovetheoverallperformance,wemadethelate
fusionofmodelsandevaluatedthetestsetthroughfivesub
missions.Thefusionwasconsideredbasedonthefollowing
569
ICMI’19,October14–18,2019,Suzhou,China EmotiWGrandChallenge
0 1 2 3
20
40
60
80
5
35
81
28
4
10
19 15
Levelsofengagement
Numberofsamples
Trainingset
Validationset
Figure3:Theclasswisedistributionoftrainingandvalida
tiondata.
Table2:Thedetailsofregressionmodels.Dimensionsare
theoutputshapeofFC,LSTMlayersinA1.A2 fromtopto
bottom.
InputModelNetwork Dimension ValMSE
F1 M1 A1 ℓ1×[128,128,128,128,1] 0.0572
F2 M2 A2 ℓ1×[100,128,128,48,128,1] 0.0575
F3 M3 A1 ℓ2×[64,128,64,128,1] 0.0585
F4 M4 A2 ℓ2×[64,64,128,48,64,1] 0.0636
equationexceptforthesecondsubmission.
Vfused=
m
k=1
αkVk, w.r.t
m
k=1
αk=1 (1)
whereVkdenotestheoutputofmodelk. Insubmission2,
wedeployedaSupportVectorRegression(SVR)withRBF
kernel for thefusion.Weadjustedtheparametersofen
semblemodelsonthevalidationset.Thecomponentsof
eachmodelwereshownin4.Basedontheresultsfromthe
Model	
1
Model	
2
Model	
3
Model	
4
Ensemble	1	(Subs	1)
Ensemble	2	(Subs	2)
Ensemble	5
(Subs	5)
Ensemble	4	(Subs	4)
Ensemble	3	(Subs	3)
Figure4:Thecomponentsofourensemblemodels.
validationset,asshownin2,weconductedthefirstthree
submissionswiththeensembleoftwo,three,andfoursingle
models.Withthedifferenceinthefeaturerepresentation,
thecombinationoftwoA1got696×10−4,thebestMSEon
thesesubmissions.Theclass-wiseMSEhasalargegapbe
tweenthelevel0andtheotherscausedbytheimbalancein
thedevelopmentset.Atthe4thattempt,wemadetheen
sembleofresultsfromsubmissionsecondandthird,which
createdbydifferentstrategies.Theyimprovedtheoverall
MSEaswellastheerroronlevel0.Inthefinalsubmission,
wefusedalltheprevioussubmissiontoobtainthefinalresult
whichachievedthebestresultwiththeMSEof0.0597and
thesmallgapbetweenclass-wiseMSE.3summarizedour
resultsoffivesubmissionsinthevalidationsetandtestset.
TheMSEforeachengagementlevelalsoprovidedincluding
disengaged(DE,0),barelyengaged(BE,1),engaged(E,2)and
highlyengaged(HE,3).
Table3:Thedetailsofsubmissions.
Subs ValMSE TestMSE
Overall DE BE E HE Overall
1 0.0497 0.3342 0.0834 0.0133 0.0660 0.0787
2 0.0372 0.3289 0.1087 0.0270 0.0353 0.0911
3 0.0518 0.2686 0.0644 0.0231 0.0640 0.0696
4 0.0777 0.2204 0.0405 0.0320 0.1022 0.0628
5 0.0958 0.2461 0.0297 0.0224 0.1378 0.0597
4 CONCLUSION
Inthispaper,wepresentedamethodforengagementpre
diction“inthewild”environment,asubchallengein7th
EmotionRecognitionintheWildChallenge2019.Ourap
proachtakesadvantageofthestate-of-the-artarchitectures
aswellasframeworksincomputervisiontosolvetheprob
lem.Ourresultsdemonstrateastrongeffectof thefacial
featureaswellaseyegazerelatedfeaturesinengagement
prediction.Withtheseeffective,wegotthebestperformance
inthechallengewithMSEof0.0597.Besidesthat,futurere
searchcouldbecontinuedtoexploreintheotherfieldssuch
asengagementinhealthcare,socialinteraction.
ACKNOWLEDGMENTS
ThisresearchwassupportedbyBasicScienceResearchPro
gramthroughtheNationalResearchFoundationofKorea
(NRF)fundedbytheMinistryofEducation(NRF-2017R1A4A
1015559,NRF-2018R1D1A3A03000947).
REFERENCES
[1] TadasBaltrušaitis,MarwaMahmoud,andPeterRobinson.2015.Cross
datasetlearningandperson-specificnormalisationforautomaticac
tionunitdetection. In201511thIEEEInternationalConferenceand
WorkshopsonAutomaticFaceandGestureRecognition(FG),Vol. 6.
IEEE,1–6.
[2] TadasBaltrusaitis,PeterRobinson,andLouis-PhilippeMorency.2013.
Constrainedlocalneuralfieldsforrobustfaciallandmarkdetection
570
EIP with Facial Behavior Features
ICMI ’19, October 14–18, 2019, Suzhou, China
in the wild. In Proceedings of the IEEE International Conference on
Computer Vision Workshops. 354–361.
[3] Tadas Baltrusaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe
Morency. 2018. Openface 2.0: Facial behavior analysis toolkit. In
2018 13th IEEE International Conference on Automatic Face & Gesture
Recognition (FG 2018). IEEE, 59–66.
[4] Nigel Bosch, Sidney K D’Mello, Ryan S Baker, Jaclyn Ocumpaugh,
Valerie Shute, Matthew Ventura, Lubin Wang, and Weinan Zhao. 2016.
Detecting Student Emotions in Computer-Enabled Classrooms.. In
IJCAI. 4125–4129.
[5] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisser
man. 2018. Vggface2: A dataset for recognising faces across pose and
age. In 2018 13th IEEE International Conference on Automatic Face &
Gesture Recognition (FG 2018). IEEE, 67–74.
[6] Cheng Chang, Cheng Zhang, Lei Chen, and Yang Liu. 2018. An Ensem
ble Model Using Face and Body Tracking for Engagement Detection.
In Proceedings of the 2018 on International Conference on Multimodal
Interaction. ACM, 616–622.
[7] Abhinav Dhall, Roland Goecke, Shreya Ghosh, and Tom Gedeon. 2019.
EmotiW 2019: Automatic Emotion, Engagement and Cohesion Predic
tion Tasks. In Proceedings of the 2019 on International Conference on
Multimodal Interaction. ACM, in press.
[8] Jennifer A Fredricks, Phyllis C Blumenfeld, and Alison H Paris. 2004.
School engagement: Potential of the concept, state of the evidence.
Review of educational research 74, 1 (2004), 59–109.
[9] Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao.
2016. Ms-celeb-1m: A dataset and benchmark for large-scale face
recognition. In European Conference on Computer Vision. Springer,
87–102.
[10] Jie Hu, Li Shen, and Gang Sun. 2018. Squeeze-and-excitation networks.
In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition. 7132–7141.
[11] Amanjot Kaur, Aamir Mustafa, Love Mehta, and Abhinav Dhall. 2018.
Prediction and localization of student engagement in the wild. In 2018
Digital Image Computing: Techniques and Applications (DICTA). IEEE,
1–8.
[12] HamedMonkaresi, Nigel Bosch, Rafael A Calvo, and Sidney K D’Mello.
2016. Automated detection of engagement using video-based estima
tion of facial expressions and heart rate. IEEE Transactions on Affective
Computing 8, 1 (2016), 15–28.
[13] Xuesong Niu, Hu Han, Jiabei Zeng, Xuran Sun, Shiguang Shan, Yan
Huang, Songfan Yang, and Xilin Chen. 2018. Automatic engagement
prediction with GAP feature. In Proceedings of the 2018 on International
Conference on Multimodal Interaction. ACM, 599–603.
[14] Chinchu Thomas, Nitin Nair, and Dinesh Babu Jayagopi. 2018. Predict
ing Engagement Intensity in the Wild Using Temporal Convolutional
Network. In Proceedings of the 2018 on International Conference on
Multimodal Interaction. ACM, 604–610.
[15] Jacob Whitehill, Zewelanji Serpell, Yi-Ching Lin, Aysha Foster, and
Javier R Movellan. 2014. The faces of engagement: Automatic recogni
tion of student engagement from facial expressions. IEEE Transactions
on Affective Computing 5, 1 (2014), 86–98.
[16] Erroll Wood, Tadas Baltrusaitis, Xucong Zhang, Yusuke Sugano, Peter
Robinson, and Andreas Bulling. 2015. Rendering of eyes for eye
shape registration and gaze estimation. In Proceedings of the IEEE
International Conference on Computer Vision. 3756–3764.
[17] Jianfei Yang, Kai Wang, Xiaojiang Peng, and Yu Qiao. 2018. Deep
Recurrent Multi-instance Learning with Spatio-temporal Features for
Engagement Intensity Prediction. In Proceedings of the 2018 on Inter
national Conference on Multimodal Interaction. ACM, 594–598.
[18] Amir Zadeh, Yao Chong Lim, Tadas Baltrusaitis, and Louis-Philippe
Morency. 2017. Convolutional experts constrained local model for