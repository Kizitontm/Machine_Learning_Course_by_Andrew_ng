function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % number of centroids

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); % this corresponds to c^i

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% create a distance matrix of mxk and initialize to zeros
m = size(X,1);

% dist_mat = zeros(m, K);

% for i = 1:m
	% dists = zeros(K,1);
	% for k= 1:K
		% dists(k) = norm(X(i,:) - centroids(k,:));
	% end
	% dist_mat(i,:) = dists';
% end
% [minval, ks] = (min(dist_mat,[],2));


for i = 1:m
	% vector to store distances of current example from each centroid
	dists = zeros(K,1); 
	
	% compute distace of each centroid from current example
	for k= 1:K
		dists(k) = norm(X(i,:) - centroids(k,:));
	end
	
	% get the index (centroid, k) of minimum distance
	[min_dist k] = min(dists);
	
	% store in the vector of closest centroids
	% This is cluster assignment
	idx(i) = k;
end

% =============================================================

end

