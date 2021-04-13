function [nrx_timeline_merged, time_merged] = helper_merge_timeline(merge_cnt, nrx_timeline, time)

size_nrx_timeline = size(nrx_timeline, 2);

new_t_size = round(size_nrx_timeline/merge_cnt);

nrx_timeline_merged = sum( reshape(nrx_timeline, [merge_cnt, new_t_size]) );

time_merged = reshape(time, [merge_cnt, new_t_size]);
time_merged = time_merged(1,:);

end

