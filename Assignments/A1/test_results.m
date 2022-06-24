load naive_errs.mat
load gaussian_errors.mat


gaussian_errors_percentage = errs ./ 400;
naive_errors_percentage = naive_errs ./ 400;

gaussian_errors_total = sum(errs) / 4000;
naive_errors_total = sum(naive_errs) / 4000;