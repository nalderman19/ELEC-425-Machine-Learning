% CMPE 425 - Assignment 1
% Nicholas Alderman - 20060982 - 16naa5
% October 12, 2021

load a1digits.mat

%2 Training Naive Bayes Classifiers
% convert training data to binary values with threshold using fix
digits_train_binary = (digits_train>0.5)

% now get eta_ki = p(b_i=1|C_k)
eta = sum(digits_train_binary(:,:,:),2)./ 700
m_eta = 1-eta

% display results in subplot
for i = 1:10
   subplot(2,5,i)
   imagesc(reshape(eta(:,1,i),8,8)'); axis equal; axis off; colormap gray;
end

% 3 Test Performance
% naive bayes p(C_k|x) = p(b|C_k,eta) * p(C_k) = Prod(eta * (1 - eta)) *
% 1/10
digits_test_binary = (digits_test>0.5)

% if feature value is 1, use eta, if zero use 1-eta
for i = 1:10
    % convert data_test_binary matrix to contain only values eta or 1-eta
    eta_combined = m_eta(:,1,i) .* (digits_test_binary==0)
    eta_combined_p = eta(:,1,i) .* (digits_test_binary==1)
    eta_combined(eta_combined == 0) = eta_combined_p(eta_combined==0)
    temp = reshape(prod(eta_combined),1,400,10)
    naive_test(i,:,:) = temp
end

% normalize so that each point sums to 1
naive_test(:,:,:) = naive_test(:,:,:)./sum(naive_test)

for i = 1:10
    [mx, idx] = max(naive_test(:,:,i), [], 1)
    naive_errs(i) = nnz(idx - i)
end

naive_errors_total = (sum(naive_errs) / 4000) * 100;
disp("The total error rate for naive bayes classifier is:")
fprintf('Percent Error: %0.3f %%', naive_errors_total);