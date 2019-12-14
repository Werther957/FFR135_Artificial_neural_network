T = 10;
Iter = 10^7;

Layer = 3;
layer1_no = 20;
layer2_no = 20;

Learn_rate = 0.02;


traning = csvread('training_set.csv',0,0);
Input_M = traning(:,1:2);
target_sel = traning(:,3);

Validation =  csvread('validation_set.csv',0,0);


%target_sel = D;

   %target_sel = test_table(i,:)
%     disp('The input index:');
%     disp(i);
    for testtimes = 1:T
        disp('The input index:');
        disp(testtimes);
        Weights_M1 = -0.5*ones(2,layer1_no) + 1*rand(2,layer1_no);
        Weights_M2 = -0.5*ones(layer1_no,layer2_no) + 1*rand(layer1_no,layer2_no);
        Weights_M3 = -0.5*ones(layer2_no,1) + 1*rand(layer2_no,1);
        
        Threshold1 = zeros(1,layer1_no);
        Threshold2 = zeros(1,layer2_no);
        Threshold3 = 0;
        
%         Threshold1 = -1*ones(1,layer1_no) + 2*rand(1,layer1_no);
%         Threshold2 = -1*ones(1,layer2_no) + 2*rand(1,layer2_no);
%         Threshold3 = -1 + 2*rand(1);
    
        linear_status = false;
        for iter_index = 1:Iter
            %use a random input as required:  stochastic gradient descent
            rand_index = randi([1,10000],1,1);
            input_random = Input_M(rand_index,:);
            
            %%%%%%%%%%%%%%%%%% output

            V1 = tanh(-Threshold1+input_random*Weights_M1);
            V2 = tanh(-Threshold2+V1*Weights_M2);
            Out = tanh(-Threshold3+V2*Weights_M3);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            deriv_out = 1-(tanh(-Threshold3+V2*Weights_M3).^2);
            error_out = (target_sel(rand_index)-Out)*deriv_out;
            
            deriv_2 =  1-(tanh(-Threshold2+V1*Weights_M2).^2);
            error2 = error_out*Weights_M3'.*deriv_2;
     %%%%%%????       
            deriv_1 =  1-(tanh(-Threshold1+input_random*Weights_M1).^2);
            error1 = error2*Weights_M2'.*deriv_1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            

            Weights_M1 = Weights_M1 + Learn_rate*(input_random' * error1);
            Threshold1 = Threshold1 - Learn_rate*error1;
            Weights_M2 = Weights_M2 + Learn_rate*(V1' * error2);
            Threshold2 = Threshold2 - Learn_rate*error2;
            Weights_M3 = Weights_M3 + Learn_rate*(V2' * error_out );
            Threshold3 = Threshold3 - Learn_rate*error_out;
            
            %%%%%%%%%%%

            if(rem(iter_index,20)==0 && check_linearity(Weights_M1,Weights_M2,Weights_M3,Threshold1,Threshold2,Threshold3,Validation))
                linear_status = true;
                disp('OK');
                break;
            end
        end

        if(linear_status)
            break;
        end

    end
        if(~linear_status)
           disp('No OK'); 
        end




function [flag_linearity] = check_linearity(Weights_M1,Weights_M2,Weights_M3,Threshold1,Threshold2,Threshold3,Validation)
    persistent C_out;
    flag_linearity = false;
    if isempty(C_out)
        C_out = 1;
    end

    
    for index = 1:5000        
      V1_check = tanh(-Threshold1+Validation(index,1:2)*Weights_M1);
      V2_check = tanh(-Threshold2+V1_check*Weights_M2);
      Out_check(index) = tanh(-Threshold3+V2_check*Weights_M3);

    end
    Out_check(Out_check >= 0) = 1;
    Out_check(Out_check < 0) = -1;
    targettemp = Validation(:,3);

    C = 1/5000/2* sum(abs(Out_check - targettemp'),'all');
    if (C < C_out)
        C_out = C
    end 

    
    if (C <= 0.12)
        flag_linearity = true;
        csvwrite('w1.csv',Weights_M1');
        csvwrite('w2.csv',Weights_M2');
        csvwrite('w3.csv',Weights_M3');
        csvwrite('t1.csv',Threshold1');
        csvwrite('t2.csv',Threshold2');
        csvwrite('t3.csv',Threshold3');
        
    end

end

