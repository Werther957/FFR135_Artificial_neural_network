Iter = 20;

Learn_rate = 0.1;
exerciseNumber = 1;

[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(exerciseNumber);

m_XTrain = mean(xTrain,2);

xTrain = xTrain - m_XTrain;
xValid = xValid - m_XTrain;
xTest  = xTest - m_XTrain;

n_testinput = size(xTrain,2);
batch_size = 100;

%%% Start Training
for nn_number = 1:4
    %init the 4 network in order
    switch nn_number
        case 1
            L = 1;
            [weights,thresholds] = Init_weights(3072,10,L,0);
        case 2
            L = 2;
            [weights,thresholds] = Init_weights(3072,10,L,10);
        case 3
            L = 2;
            [weights,thresholds] = Init_weights(3072,10,L,50);
        case 4
            L = 3;
            [weights,thresholds] = Init_weights(3072,10,L,50);
    
    end
    
   %initialize the weights and thresholds at each epoch
    weights_temp = weights;
    thresholds_temp = thresholds; 

    for epoch=1:20

        p = randperm(n_testinput); % shuffle the index for all the test vectors


        for batch_index = 1:n_testinput

            index = p(batch_index); %index to get the input from the dataset
            [output,b_array] = forward(xTrain(:,index),weights,thresholds,L);
            output{L};
            [error_array] = backward(output,weights,thresholds,L,tTrain(:,index),b_array);
            [weights_temp,thresholds_temp] = update(xTrain(:,index),output,weights_temp,thresholds_temp,L,error_array,Learn_rate);
                 %for debug update like scd     

            if(0 == rem(batch_index,batch_size))
               %update the weight and threshold, the new para work in the next
               %batch
               weights = weights_temp;
               thresholds = thresholds_temp;  

            end

        end
        %do test with new weights after one epoch
        % [cerror, ~] = test(weights, thresholds, xValid, tValid)
        cerror = classifiError(tValid,xValid,weights,thresholds,L)
        cerrorValid{nn_number}{epoch} = cerror
        %cerror = classifiError(tTest,xTest,weights,thresholds,L)
        %cerrorTest{nn_number}{epoch} = cerror
        cerror = classifiError(tTrain,xTrain,weights,thresholds,L)
        cerrorTrain{nn_number}{epoch} = cerror
         
        storeWeight{nn_number}{epoch} = weights;
        storethreshold{nn_number}{epoch} = thresholds;
       % weights{1}

    end
end


%% 
%Get the result for report
%find the min error in each nn

    [minerror,minindex] = min(cell2mat(cerrorValid{1}));
    %calculate the test error
    cerror = classifiError(tTest,xTest,storeWeight{1}{minindex},storethreshold{1}{minindex},1)
    disp(['network1  ','epoch:',num2str(minindex),'   ValidError:',num2str(minerror),'   TrainError',num2str(cerrorTrain{1}{minindex}),'   TestError',num2str(cerror)]);
    
    [minerror,minindex] = min(cell2mat(cerrorValid{2}));
    %calculate the test error
    cerror = classifiError(tTest,xTest,storeWeight{2}{minindex},storethreshold{2}{minindex},1);
     disp(['network2  ','epoch:',num2str(minindex),'   ValidError:',num2str(minerror),'   TrainError',num2str(cerrorTrain{2}{minindex}),'   TestError',num2str(cerror)]);

    [minerror,minindex] = min(cell2mat(cerrorValid{3}));
    %calculate the test error
    cerror = classifiError(tTest,xTest,storeWeight{3}{minindex},storethreshold{3}{minindex},2);
     disp(['network3  ','epoch:',num2str(minindex),'   ValidError:',num2str(minerror),'   TrainError',num2str(cerrorTrain{3}{minindex}),'   TestError',num2str(cerror)]);
    
    [minerror,minindex] = min(cell2mat(cerrorValid{4}));
    %calculate the test error
    cerror = classifiError(tTest,xTest,storeWeight{4}{minindex},storethreshold{4}{minindex},3);
     disp(['network4  ','epoch:',num2str(minindex),'   ValidError:',num2str(minerror),'   TrainError',num2str(cerrorTrain{4}{minindex}),'   TestError',num2str(cerror)]);

     
     
 %%%% plot

epochs = linspace(1,20,20);
semilogy(epochs, cell2mat(cerrorValid{1})); hold on
semilogy(epochs,  cell2mat(cerrorValid{2}));
semilogy(epochs,  cell2mat(cerrorValid{3})); 
semilogy(epochs,  cell2mat(cerrorValid{4}));

semilogy(epochs, cell2mat(cerrorTrain{1})); 
semilogy(epochs,  cell2mat(cerrorTrain{2}));
semilogy(epochs,  cell2mat(cerrorTrain{3})); 
semilogy(epochs,  cell2mat(cerrorTrain{4}));


legend('Net 1 Val', 'Net 2 Val', 'Net 3 Val', 'Net 4 Val', 'Net 1 Train', 'Net 2 Train', 'Net 3 Train', 'Net 4 Train','Location', 'Northeast', 'fontsize',11);

title('Training&Validation errors per epoch')
xlabel('Epoch')
ylabel('Classification error')
%%











function [cerror] = classifiError(targets,inputs,weights,thresholds,L)
    n=size(targets,2);
    cerror=0;
    for i=1:n
       [output,~]=forward(inputs(:,i),weights,thresholds,L);
       
       [~, index] = max(output{L});
       
       output_temp = zeros(size(output{L}),'like',output{L});
       
       output_temp(index) = 1;
  
       cerror=cerror+1/(2*n)*sum(abs(targets(:,i)-output_temp));
    end
    
end  
    


%for each input
function [weightst,thresholdt] = update(input,output,weights,threshold,L,error_array,rate)
    for l=1:L

        if(1==l)
%             rate = rate
%             size(error_array{l})
%             size(input)


            weightst{l} = weights{l} + rate.*error_array{l}*input';           
        else

            weightst{l} = weights{l} + rate.*error_array{l}*output{l-1}';
        end
        thresholdt{l} = threshold{l} - rate.*error_array{l};
    end


end



function [error_array] = backward(output,weights,threshold,L,target,b_array)
    error_array = {};

    for l = L:-1:1

       if(L == l)
         % error_array{l} = sigmf(b_array{l},[1 0]).*(1-sigmf(b_array{l},[1 0])).*(target - output{l});
         error_array{l} = output{l}.*(1-output{l}).*(target - output{l});
           
       else
          % error_array{l} = weights{l+1}'*error_array{l+1}.*sigmf(b_array{l},[1 0]).*(1-sigmf(b_array{l},[1 0])); 
           error_array{l} = weights{l+1}'*error_array{l+1}.*output{l}.*(1-output{l});
       end
                     
    end
    
end

function [output,b_array] = forward(input,weights,threshold,L)
    %tmp_out = {};
    %layers = lenght(weights);
  
    for l = 1:L
       if(1 == l)

           b = weights{l}*input - threshold{l};
       else
            %size(weights{l})
            %size(threshold{l})
           b = weights{l}*output{l-1} - threshold{l};         
       end
       
       output{l} = sigmf(b,[1 0]);
       b_array{l} = b;
       
      % if(l == L)
       %    output{l+1} = sigmf(b,[1 0]) %store the final output result
      % end
                     
    end

end


function [weights,thresholds] = Init_weights(in_N,out_N,L,h_N)

    a = in_N;
    weightsl={};
    weightslL = [];
    
    thresholdl={};
    thresholdlL = [];
    
    for l=1:L
       if(L == 1)
                   
        b = out_N;

        tmp_weigh_n = in_N;
        tmp = 1/sqrt(tmp_weigh_n);

        weightsl0 = normrnd(0,tmp,b,a);
        thresholdl0 = zeros(b,1);
           
       else
           if(l == 1)
                b = h_N;

                tmp_weigh_n = in_N;
                tmp = 1/sqrt(tmp_weigh_n);

                weightsl0 = normrnd(0,tmp,b,a);
                thresholdl0 = zeros(b,1);
           else
               if(l == L)
                b = h_N;
   
                tmp_weigh_n = h_N;
                tmp = 1/sqrt(tmp_weigh_n);

                weightslL = normrnd(0,tmp,out_N,b);
                thresholdlL = zeros(out_N,1);
                   
               else
                   tmp_weigh_n = h_N;
                   tmp = 1/sqrt(tmp_weigh_n);
                   weightsl{l-1} =  normrnd(0,tmp,h_N,h_N); %  reach here only when the l >= 2 
                   thresholdl{l-1} = zeros(h_N,1);
               end
               
               
               
           end
           
           
       end
    end
    
     weights = {weightsl0,weightsl{:},weightslL};
     size(weights)
     thresholds = {thresholdl0,thresholdl{:},thresholdlL};


end

function [result] = sigmoid(b)
    result = 1./(1+exp(-b));
end
