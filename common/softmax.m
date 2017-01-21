% h for hyppothesis
function h = softmax(x)

h = bsxfun(@minus, x, max(x)); % to prevent overflows
h = exp(h);
h = bsxfun(@rdivide, h, sum(h)); % numClasses x numSamples
  
end

