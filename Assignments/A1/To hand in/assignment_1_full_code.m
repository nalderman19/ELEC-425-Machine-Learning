

load a1digits.mat

% 1.2 Training gaussian
% p(C_k) = a_k
% p(x|C_k) = (2pi*sigma^2)^(-D/2) * exp{-(1/2*sigma^2 * sum(x_i-mu_ki)^2}
% use these two values to calculate p(C_k|x)

% mu_ki = sum of each vector of D features divided by the number of
% training data points
% mu = average value for each feature in a class
mle = [];

for j = 1:10
    for i = 1:64
        mle(i,j) = sum(digits_train(i,:,j))/700
    end
end
% now have mu array for all 10 digits, need to display them with subplot &

% fix shape of mle
mle2 = reshape(mle,64,1,10)
mle2 = repmat(mle2,1,700,1)

% now need to get sigma^2
% sum of difference between feature and mle across all classes across all 
% data points across all features
% x = digits_X(i,j,k) -- j = data point #, i = feature number, k = class

s2 = sum((digits_train - mle2).^2, 'all') / (64 * 7000);

for i = 1:10
   subplot(2,5,i)
   imagesc(reshape(mle(:,i),8,8)'); axis equal; axis off; colormap gray;
end

% obtain and plot standard deviation: sigma = sqrt(sigma^2)
subplot(2,5,1)
sd = sqrt(s2);
text(10, 10, "The pixel noise standard deviation is:")
text(10,12, sprintf('%.6f',sd))

%2 Training Naive Bayes Classifiers
% convert training data to binary values with threshold using fix
digits_train_binary = (digits_train>0.5)

% now get eta_ki = p(b_i=1|C_k)
eta = sum(digits_train_binary(:,:,:),2)./ 700
m_eta = 1-eta

% display results in subplot
% for i = 1:10
%    subplot(2,5,i)
%    imagesc(reshape(eta(:,1,i),8,8)'); axis equal; axis off; colormap gray;
% end

% 3 Test Performance
% gaussian test - p(C_k|x) = p(x|C_k)*p(C_k)
t1 = (2*pi*s2)^-32
t2 = (-1/(2*s2)) 

for i = 1:10
    gaussian_test(i,:,:) = ((t1 .* exp(t2 .* sum((digits_test(:,:,:) - mle(:,i)).^2))).*(1/10));
    % bayes theorem
end

% normalize so that each data point sums to 1
gaussian_test(:,:,:) = gaussian_test(:,:,:)./sum(gaussian_test)

% select most likely class for each data point
for i = 1:10
    [mx, idx] = max(gaussian_test(:,:,i), [], 1)
    gaussian_errs(i) = nnz(idx - i)
end

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

gaussian_errors_total = sum(gaussian_errs) / 4000;
naive_errors_total = sum(naive_errs) / 4000;