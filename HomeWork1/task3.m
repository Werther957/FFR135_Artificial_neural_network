T = 2*10^5;
P = 45;
N = 200;
B = 2;
EXE_TIMES = 100;


mypara = zeros(1,EXE_TIMES);

for exetime = 1:EXE_TIMES
    Pattern_Matrix = GenPatternMatrix(P,N);
    Weight_M = GenWeightMatrix(Pattern_Matrix,N);
    Weight_M = Weight_M - diag(diag(Weight_M)); % diag ---> 0
    Pattern_Matrix_update = Pattern_Matrix(1,:);
    sum1 = zeros(1,T);
    for testtime = 1:T
        temp = rand;
        index_neuron = randi([1,N],1);
        b = Pattern_Matrix_update * Weight_M(index_neuron, :)';
        g = 1/(1+exp(-2*B*b));
        if(temp > g)
            result = -1;
        else
            result = 1;
        end
        %Pattern_Matrix_update(index_neuron)
        %result
        if(result ~= Pattern_Matrix_update(index_neuron))
            Pattern_Matrix_update(index_neuron) = result;
        end

        sum1(testtime) = Pattern_Matrix_update * Pattern_Matrix(1,:)'/N;

    end
        mypara(exetime) = sum(sum1,'all')/T;
        text = ['The result at execution time',num2str(exetime),':   ',num2str(mypara(exetime))];
        disp(text);

    
end    
    
Average_Time = sum(mypara, 'all')/EXE_TIMES    




function [Pattern_Matrix] = GenPatternMatrix(pattern_num,bits_num)


    Pattern_Matrix = randi([0,1],[pattern_num,bits_num]);   
    Pattern_Matrix(Pattern_Matrix == 0) = -1;


end

function [Weight_M] = GenWeightMatrix(pattern,pattern_num)

        Weight_M = pattern'*pattern/pattern_num;
end
