
clc; clear all; close all;

facedir_training = 'training\';
facedir_testing = 'testing2\';

colorSpace = 'Gradient'; %RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient
nSubject = 33; %Number of persons in the training dataset. 
K =60; %the number of eigen vectors
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read in all the images
allFaceIms_training = [];
labels = []; %each row represents [folder, image in the folder] i.e., [person, which image of the person].
for iSubject = 1:nSubject
    [FaceIms,nrows,ncols,np] = getAllIms(sprintf('%s%02d\\',facedir_training,iSubject),colorSpace);
    if isempty(FaceIms), continue; end
    allFaceIms_training = [allFaceIms_training; FaceIms];    
    labels = [labels; [iSubject*ones(size(FaceIms,1),1) (1:size(FaceIms,1))']]; 
end
allFaceIms_training = allFaceIms_training'; %every column in allFaceIms_training is one face image

%compute Mean

allFaceIms_training_Mean= mean(allFaceIms_training,2);

% subtract mean from the images

 allFaceIms_training =allFaceIms_training-allFaceIms_training_Mean;
 
% Calculating Eigen values

Cov2=allFaceIms_training'*allFaceIms_training;

[V,egValues]=eig(Cov2);

temp_U=allFaceIms_training*V;

% taking k largest eigen vectors

U=temp_U( : ,(size(temp_U,2)-(K-1)):end);

Alpha=[];

for j=1:size(allFaceIms_training,2) 
    for i=K:-1:1
        Alpha(j,i)= U(:,i)'*allFaceIms_training(:,j);
    end
end    
Ref_Feature=Alpha';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read in all the images
allFaceIms_testing = [];
labels1 = []; %each row represents [folder, image in the folder] i.e., [person, which image of the person].
for iSubject = 1:nSubject
    [FaceIms,nrows,ncols,np] = getAllIms(sprintf('%s%02d\\',facedir_testing,iSubject),colorSpace);
    if isempty(FaceIms), continue; end
    allFaceIms_testing = [allFaceIms_testing; FaceIms];    
    labels1 = [labels1; [iSubject*ones(size(FaceIms,1),1) (1:size(FaceIms,1))']]; 
end
allFaceIms_testing = allFaceIms_testing'; %every column in allFaceIms_training is one face image

%compute Mean

allFaceIms_testing_Mean= mean(allFaceIms_testing,2);

% subtract training mean from the images

 %allFaceIms_testing =allFaceIms_testing-allFaceIms_testing_Mean;
 allFaceIms_testing =allFaceIms_testing-allFaceIms_training_Mean;
 
Alpha_test=[];
% use eigen vectors from training data and projecting testing images on that sub space

for j=1:size(allFaceIms_testing,2) 
    for i=K:-1:1
        %Alpha_test(j,i)= R(:,i)'*allFaceIms_testing(:,j);
         Alpha_test(j,i)= U(:,i)'*allFaceIms_testing(:,j);
    end
end    
Test_Feature=Alpha_test';

D= zeros(size(Ref_Feature,2),size(Test_Feature,2));

% calculating Euclidean, Norm1, Norm2 and Manhattan(City Block) distance.
% Edit D(j,i)=dist1/dist2/dist3/dist4 accordingly. 


for i=1:size(Test_Feature,2) 
   for j=1:size(Ref_Feature,2)
        % dist1 is euclidean distance
        dist1 = sqrt(sum((Test_Feature(:,i)-Ref_Feature(:,j)).^2));
        % difference in norm1 distance
        dist2 = abs(norm(Test_Feature(:,i),1)-norm(Ref_Feature(:,j),1));
        % difference in norm2 distance
        dist3 = abs(norm(Test_Feature(:,i))-norm(Ref_Feature(:,j)));
        % city block (manhattan) distance
        dist4 = sum(abs(Test_Feature(:,i)-Ref_Feature(:,j)));
        D(j,i) = dist4; % Edit D(j,i)=dist1/dist2/dist3/dist4 accordingly. 
    end
end


D2 = D;
% get min value and the row where the min value occurs
% which corresponds to image index in training dataset
[minValue, rowMinValue] = min(D2,[],1);
% construct matrix with these values
results2 = [minValue;rowMinValue];
% get subject and view from training set according the image index
% from the model (rowMinValue) and add it to results matrix
results2 = [results2;labels(rowMinValue,:)'];
% add the actual lables for the test images
results2 = [results2;labels1'];
results2 = results2';
% test accuracy 
imageAccu = sum(results2(:,3)==results2(:,5)&results2(:,4)==results2(:,6))/size(allFaceIms_testing,2);
disp(['accuracy of images ',num2str(imageAccu*100),' %'])

subjAccu = sum(results2(:,3)==results2(:,5))/size(allFaceIms_testing,2);
disp(['accuracy of subjects ',num2str(subjAccu*100),' %'])




     
























