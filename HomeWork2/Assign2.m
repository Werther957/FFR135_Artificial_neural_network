T = 10;
Iter = 10^5;
N = 4; %petron


Learn_rate = 0.02;


Input_M = csvread('input_data_numeric.csv',0,1);

D = [1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1];
B = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1];
A = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1];
F = [1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1];
C = [1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
E = [-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1];



test_table = [D; B; A; F; C; E]

%target_sel = D;
for i=1:size(test_table)
    target_sel = test_table(i,:)
    disp('The input index:');
    disp(i);
    for testtimes = 1:T
        Weights_M = -0.2+0.4*rand(4,1);
        Threshold = -1+2*rand;
    
        linear_status = false;
        for iter_index = 1:Iter
            %use a random input as required:  stochastic gradient descent
            rand_index = randi([1,16],1,1);
            input_random = Input_M(rand_index,:);
            output = tanh(0.5*(-Threshold+input_random*Weights_M));
            deriv = 1-tanh(0.5*(-Threshold+input_random*Weights_M))^2;
            error = (target_sel(rand_index)-output)*deriv*0.5;

            Weights_M = Weights_M + Learn_rate*(error*input_random');
            Threshold = Threshold - Learn_rate*error;


            if(check_linearity(Weights_M,Threshold,target_sel,Input_M))
                linear_status = true;
                disp('linear');
                break;
            end
        end

        if(linear_status)
            break;
        end

    end
        if(~linear_status)
           disp('Non linear'); 
        end


end

function [flag_linearity] = check_linearity(Weights_M,Threshold,target_sel,Input_M)
    flag_linearity = false;
    for index = 1:16
        tempout(index) = tanh(0.5*(-Threshold + Input_M(index,:)*Weights_M));
    end
    tempout(tempout >= 0) = 1;
    tempout(tempout < 0) = -1;
    
    if (tempout == target_sel)
        flag_linearity = true;
    end

end







