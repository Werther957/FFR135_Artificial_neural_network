Pattern_Vect = [12,24,48,70,100,120]  
%Pattern_Vect = [37]  %used to check the one step p when a= Pattern_Vect/Pattern_Bits is around 0.185
%Pattern_Bits = 200;
Pattern_Bits = 120;
TRIALS_Times = 10^5;


for i = 1:length(Pattern_Vect) 
   % error_counter = 0;
    pattern_num = Pattern_Vect(i);

    error_counter = Errors_Count(Pattern_Bits,pattern_num,TRIALS_Times);

    fprintf('One-step error probability : %d\t %d\n', [pattern_num error_counter/TRIALS_Times]);

end


function [error_counter] = Errors_Count(Pattern_Bits,pattern_num,TRIALS_Times)

    error_counter = 0;

    
    
    for t = 1:TRIALS_Times
        
        %Generate new pattern each trial
        Pattern_Matrix = GenPatternMatrix(pattern_num,Pattern_Bits);
        Weight_M = GenWeightMatrix(Pattern_Matrix,pattern_num);
       % Weight_M = Weight_M - diag(diag(Weight_M)); % diag ---> 0
        

        index_pattern = randi([1,pattern_num],1);    
        index_neuron = randi([1,Pattern_Bits],1);

 
        result = Pattern_Matrix(index_pattern,:)*Weight_M(index_neuron, :)';

        if(result >= 0)
            mystate = 1;
        else
            mystate = -1;
        end

        if(mystate ~= Pattern_Matrix(index_pattern,index_neuron))
            error_counter = error_counter + 1;
        end
    end
end



function [Pattern_Matrix] = GenPatternMatrix(pattern_num,bits_num)


    Pattern_Matrix = randi([0,1],[pattern_num,bits_num]);   
    Pattern_Matrix(Pattern_Matrix == 0) = -1;


end


function [Weight_M] = GenWeightMatrix(pattern,pattern_num)

        Weight_M = pattern'*pattern/pattern_num;
end















