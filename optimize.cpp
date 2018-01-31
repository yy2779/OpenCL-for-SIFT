// optimized kernel
// process the image with Gaussian Blur
__kernel void GaussianBlur(__global unsigned int* img,
                           __global unsigned int* dst,
                           __local int* local_src,
                           const int local_src_x,
                           const int local_src_y,
                           __local float* mask,
                           const float sigma,
                           __local unsigned int* mask_sum,
                           const unsigned int img_rows,
                           const unsigned int img_cols)

// img: input, a 2D matrix, one channel/gray image, the base image for GaussianBlur
// dst: output, a 2D matrix, the blurred image
// mask_sum: mask_sum[0] = 0
// sigma: a constant number, coefficient for Gaussian
// img_rows: a constant number, img.shape[0]
// img_cols: a constant number, img.shape[1]
// http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing.php
{
	int gi = get_global_id(0);
	int gj = get_global_id(1);
	int li = get_local_id(0);
	int lj = get_local_id(1);

	const int size = (float)(sigma * 6 + 1) <= (int)(sigma * 6 + 1) + 0.4? (int)(sigma * 6 + 1) : (int)(sigma * 6 + 1) + 1;
	int append_number = (size - 1) / 2;
	int li_size = local_src_x + 1 - size;  // The size of work group in x direction
	int lj_size = local_src_y + 1 - size;	// The size of work group in y direction
    // Copy value to local memory
	local_src[(li + append_number)* local_src_y + lj + append_number] = img[gi * img_cols + gj];
	// deal with the last three columns and rows
	int new_pos_x;
	int new_pos_g_x;
	int new_pos_y;
	int new_pos_g_y;

	if(li >= li_size - append_number || lj >= lj_size - append_number || li < append_number || lj < append_number)
	{	
		new_pos_x = append_number + li;
		new_pos_y = append_number + lj;
		new_pos_g_x = gi;
		new_pos_g_y = gj;

		if(li >= li_size - append_number) 
		{
			new_pos_x = li + append_number + 2 * (li_size - li) - 1;
			new_pos_g_x = gi + 2 * (li_size - li) - 1;
			if(new_pos_g_x >= img_rows)
				new_pos_g_x = gi;
		}

		else if(li < append_number)
		{
			new_pos_x = append_number - 1 - li;		// append_number + append_number - 1 - li
			new_pos_g_x = gi - 2 * li - 1;
			if(new_pos_g_x < 0)
				new_pos_g_x = gi;
		}

		if(lj >= lj_size - append_number) 
		{
			new_pos_y = lj  + append_number + 2 * (lj_size - lj) - 1;
			new_pos_g_y = gj + 2 * (lj_size - lj) - 1;
			if(new_pos_g_y >= img_cols)
				new_pos_g_y = gj;
		}
		else if(lj < append_number)
		{
			new_pos_y = append_number - 1 - lj;
			new_pos_g_y = gj - 2 * lj - 1;
			if(new_pos_g_y < 0)
				new_pos_g_y = gj;
		}
		if(new_pos_x != append_number + li && new_pos_y != append_number + lj)
		{
			local_src[(append_number+li) * local_src_y + new_pos_y] = img[gi * img_cols + new_pos_g_y];
			local_src[new_pos_x * local_src_y + append_number + lj] = img[new_pos_g_x * img_cols + gj];
		}
		local_src[new_pos_x * local_src_y + new_pos_y] = img[new_pos_g_x * img_cols + new_pos_g_y];
	}

	mask_sum[0] = dst[0];

	barrier(CLK_GLOBAL_MEM_FENCE);

// Build mask in local memory	
	if(li < size && lj < size)
	{
		float x = li - (size - 1) / 2;
		float y = lj - (size - 1) / 2;
		mask[li * size + lj] = exp(-(x*x + y*y) / (2 * sigma*sigma));
		int mask_float2int = (int)(mask[li * size + lj] * 10e5);
		atomic_add(&mask_sum[0], mask_float2int);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
    
	if(li < size && lj < size)
	{
		mask[li * size + lj] /= (float)(mask_sum[0]) * 10e-7;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

// Calculate dst Matrix
	dst[gi * img_cols + gj] = 0;
	float temp = 0;
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			int lx = li + i;
			int ly = lj + j;
			temp += (float)local_src[lx * local_src_y + ly] * mask[i * size + j];	
		}
	}
	dst[gi * img_cols + gj] = (int)temp;
	
}

// optimized kernel
// used to build an octave of Gaussian Pyramid, to compose one Octave of the Pyramid
__kernel void GaussianPyramid(__global unsigned int* img,
                              __global unsigned int* dst,
                              int nlayers,
                              int img_rows,
                              int img_cols,
                              __global float* sig,
                              __local float* mask,
                              __local int* mask_sum,
                              __local int* local_src,
                              int max_size,
                              int local_src_x,
                              int local_src_y) 
{
	// GPU_kernel shape: [rows, cols, nlayers]
    // img: input, a 2D matrix, one channel/gray image, the base image for GaussianPyramid
    // dst: output, a 3D matrix, should be in shape [nlayers, img_rows, img_cols]
    // sigma: input, an array
    // nlayers: input, the number of output images, NOTE nlayers = nOctaveLayers + 2 
    // img_rows, img_cols: input

	int gi = get_global_id(0);
	int gj = get_global_id(1);
	int gk = get_global_id(2);
	int li = get_local_id(0);
	int lj = get_local_id(1);
	int lk = get_local_id(2);

	float sigma = sig[gk];

	int size = (float)(sigma * 6 + 1) <= (int)(sigma * 6 + 1) + 0.4? (int)(sigma * 6 + 1) : (int)(sigma * 6 + 1) + 1;
	if(size % 2 == 0)
		size += 1;
	else
		size = size;
    
	int append_number = (size - 1) / 2;
	int li_size = local_src_x + 1 - max_size;  // The size of work group in x direction
	int lj_size = local_src_y + 1 - max_size;	// The size of work group in y direction
	if(size - 1 < li_size)
	{
        // Copy value to local memory
		local_src[(li + append_number)* local_src_y + lj + append_number] = img[gi * img_cols + gj];
		// deal with the appended columns and rows
		int new_pos_x;
		int new_pos_g_x;
		int new_pos_y;
		int new_pos_g_y;
		if(li >= li_size - append_number || lj >= lj_size - append_number || li < append_number || lj < append_number)
		{	
			new_pos_x = append_number + li;
			new_pos_y = append_number + lj;
			new_pos_g_x = gi;
			new_pos_g_y = gj;

			if(li >= li_size - append_number) 
			{
				new_pos_x = li + append_number + 2 * (li_size - li) - 1;
				new_pos_g_x = gi + 2 * (li_size - li) - 1;
				if(new_pos_g_x >= img_rows)
					new_pos_g_x = gi;
			}
			else if(li < append_number)
			{
				new_pos_x = append_number - 1 - li;		// append_number + append_number - 1 - li
				new_pos_g_x = gi - 2 * li - 1;
				if(new_pos_g_x < 0)
					new_pos_g_x = gi;
			}

			if(lj >= lj_size - append_number) 
			{
				new_pos_y = lj  + append_number + 2 * (lj_size - lj) - 1;
				new_pos_g_y = gj + 2 * (lj_size - lj) - 1;
				if(new_pos_g_y >= img_cols)
					new_pos_g_y = gj;
			}
			else if(lj < append_number)
			{
				new_pos_y = append_number - 1 - lj;
				new_pos_g_y = gj - 2 * lj - 1;
				if(new_pos_g_y < 0)
					new_pos_g_y = gj;
			}
			if(new_pos_x != append_number + li && new_pos_y != append_number + lj)
			{
				local_src[(append_number+li) * local_src_y + new_pos_y] = img[gi * img_cols + new_pos_g_y];
				local_src[new_pos_x * local_src_y + append_number + lj] = img[new_pos_g_x * img_cols + gj];
			}

			local_src[new_pos_x * local_src_y + new_pos_y] = img[new_pos_g_x * img_cols + new_pos_g_y];			
		}

		mask_sum[0] = dst[0];
		barrier(CLK_GLOBAL_MEM_FENCE);

        // Build mask in local memory	
		if(li < size && lj < size)
		{
			float x = li - (size - 1) / 2;
			float y = lj - (size - 1) / 2;
			mask[li * size + lj] = exp(-(x*x + y*y) / (2 * sigma*sigma));
			int mask_float2int = (int)(mask[li * size + lj] * 10e5);
			atomic_add(&mask_sum[0], mask_float2int);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(li < size && lj < size)
		{
			mask[li * size + lj] /= (float)(mask_sum[0]) * 10e-7;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		float temp = 0;
		for(int i = 0; i < size; i++)
		{
			for(int j = 0; j < size; j++)
			{
				int lx = li + i;
				int ly = lj + j;
				temp += (float)local_src[lx * local_src_y + ly] * mask[i * size + j];	
			}

		}
		dst[gk * img_cols * img_rows + gi * img_cols + gj] = (int)temp;
	}
	else
	{
		dst[gk * img_cols * img_rows + gi * img_cols + gj] = img[li * img_cols + lj];
	}
}



// optimized kernel
// calculate one set of different images, used to compose one Octave of the Pyramid
__kernel void DoGPyramid(__global unsigned char* img, // input, a 3D matrix, should be in shape [n, img_rows, img_cols]
                         __global int* dst, //output, a 3D matrix, should be in shape [n-1, img_rows, img_cols]
                         const unsigned int img_rows, 
                         const unsigned int img_cols) 
{
	unsigned int gj = get_global_id(0);
	unsigned int gk = get_global_id(1);
	unsigned int gi = get_global_id(2);
    unsigned int lj = get_local_id(0);
	unsigned int lk = get_local_id(1);
	unsigned int li = get_local_id(2);
    
    int idx = gi*img_rows*img_cols + gj*img_cols + gk;
    
    // store the images pieces from global memory
    __local unsigned char img_local[384];
    
    img_local[li*8*8 + lj*8 + lk] = img[idx];
    
    if(li == 4){
        img_local[(li + 1)*8*8 + lj*8 + lk] = img[idx + img_rows*img_cols];
    }
    
    // local memory barrier
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // subtract one pixel form another, cv2.subtract function
	int dst_pixel = img_local[(li + 1)*8*8 + lj*8 + lk] - img_local[li*8*8 + lj*8 + lk];
    dst[idx] = dst_pixel>0 ? dst_pixel : 0;
}


// optimized kernel
// find the local extrema
__kernel void findLocalExtrema(__global int* dogpyr, // input, DoGPyramid, a set of images, shape [nOctaveLayers+2, rows, cols]
                               __global float* kpts_list, // output, list of local extrema
                               __global int* t_idx, // used to trace the output list length
                               const unsigned int Octave, // used for output list
                               const unsigned int rows, 
                               const unsigned int cols, 
                               const float threshold, // =1, used to filter useless points
                               const unsigned int img_border) // avoid points locate on image's edge
{
	unsigned int gi = get_global_id(0);
	unsigned int gj = get_global_id(1);
	unsigned int gk = get_global_id(2);
    unsigned int lj = get_local_id(1);
	unsigned int lk = get_local_id(2);
    
    // init the global index tracer
    if(gi==0 && gj==0 && gk==0)
        t_idx[0] = 0;
    
    // init private index tracer
    int idx_atomic = -1;
    
    // used to store the images pieces, 324 = 18*18 = (16+2)*(16+2)
    __local int curr[324];
    __local int next[324];
    __local int prev[324];
    
    // avoid points locate on image's edge
    if (gj >= img_border && gj < rows - img_border) {
		if (gk >= img_border && gk < cols - img_border) {
            
            // the global index of pixels
            int idx = (gi + 1)*rows*cols + gj*cols + gk;  
            
            curr[(lj + 1) * 18 + lk + 1] = dogpyr[idx];
            next[(lj + 1) * 18 + lk + 1] = dogpyr[idx + rows*cols];
            prev[(lj + 1) * 18 + lk + 1] = dogpyr[idx - rows*cols];
            
            // process the edge pixels
            if(lj == 0){
                curr[lk+1] = dogpyr[idx - cols];
                next[lk+1] = dogpyr[idx + rows*cols - cols];
                prev[lk+1] = dogpyr[idx - rows*cols - cols];
                
                if(lk == 0){
                    curr[0] = dogpyr[idx - cols - 1];
                    next[0] = dogpyr[idx + rows*cols - cols - 1];
                    prev[0] = dogpyr[idx - rows*cols - cols - 1];
                    
                }
                else if(lk == 15){
                    curr[17] = dogpyr[idx - cols + 1];
                    next[17] = dogpyr[idx + rows*cols - cols + 1];
                    prev[17] = dogpyr[idx - rows*cols - cols + 1];
                }
            }
            if(lj == 15){
                curr[18*17 + lk+1] = dogpyr[idx + cols];
                next[18*17 + lk+1] = dogpyr[idx + rows*cols + cols];
                prev[18*17 + lk+1] = dogpyr[idx - rows*cols + cols];
                if(lk == 0){
                    curr[18*17] = dogpyr[idx + cols - 1];
                    next[18*17] = dogpyr[idx + rows*cols + cols - 1];
                    prev[18*17] = dogpyr[idx - rows*cols + cols - 1];
                }
                else if(lk==15){
                    curr[18*18-1] = dogpyr[idx + cols + 1];
                    next[18*18-1] = dogpyr[idx + rows*cols + cols + 1];
                    prev[18*18-1] = dogpyr[idx - rows*cols + cols + 1];
                }
            }
            if(lk == 0){
                curr[(lj + 1) * 18] = dogpyr[idx - 1];
                next[(lj + 1) * 18] = dogpyr[idx + rows*cols - 1];
                prev[(lj + 1) * 18] = dogpyr[idx - rows*cols - 1];
            }
            if(lk == 15){
                curr[(lj + 1) * 18 + 17] = dogpyr[idx + 1];
                next[(lj + 1) * 18 + 17] = dogpyr[idx + rows*cols + 1];
                prev[(lj + 1) * 18 + 17] = dogpyr[idx - rows*cols + 1];
            }
            
            // the local index of pixels
            int idx_local = (lj + 1) * 18 + lk + 1;
            
            // judge local extrema
			if (abs(curr[idx_local]) > threshold) {
				if (curr[idx_local] > 0  // if it's a maximum
                    && curr[idx_local] >= curr[idx_local + 1] && curr[idx_local] >= curr[idx_local - 1] 
					&& curr[idx_local] >= curr[idx_local + 18] && curr[idx_local] >= curr[idx_local - 18]
					&& curr[idx_local] >= curr[idx_local + 18 + 1] && curr[idx_local] >= curr[idx_local + 18 - 1]
					&& curr[idx_local] >= curr[idx_local - 18 + 1] && curr[idx_local] >= curr[idx_local - 18 - 1]
					&& curr[idx_local] >= next[idx_local] && curr[idx_local] >= next[idx_local + 1]
					&& curr[idx_local] >= next[idx_local - 1] && curr[idx_local] >= next[idx_local + 18]
					&& curr[idx_local] >= next[idx_local - 18] && curr[idx_local] >= next[idx_local + 18 + 1]
					&& curr[idx_local] >= next[idx_local + 18 - 1] && curr[idx_local] >= next[idx_local - 18 + 1]
					&& curr[idx_local] >= next[idx_local - 18 - 1]
					&& curr[idx_local] >= prev[idx_local] && curr[idx_local] >= prev[idx_local + 1]
					&& curr[idx_local] >= prev[idx_local - 1] && curr[idx_local] >= prev[idx_local + 18]
					&& curr[idx_local] >= prev[idx_local - 18] && curr[idx_local] >= prev[idx_local + 18 + 1]
					&& curr[idx_local] >= prev[idx_local + 18 - 1] && curr[idx_local] >= prev[idx_local - 18 + 1]
					&& curr[idx_local] >= prev[idx_local - 18 - 1])
				{
                    idx_atomic = atomic_add(&t_idx[0], 1); // get the private index for ouput
				}
				else if (curr[idx_local] < 0  // if it's a minimum
                    && curr[idx_local] <= curr[idx_local + 1] && curr[idx_local] <= curr[idx_local - 1] 
					&& curr[idx_local] <= curr[idx_local + 18] && curr[idx_local] <= curr[idx_local - 18]
					&& curr[idx_local] <= curr[idx_local + 18 + 1] && curr[idx_local] <= curr[idx_local + 18 - 1]
					&& curr[idx_local] <= curr[idx_local - 18 + 1] && curr[idx_local] <= curr[idx_local - 18 - 1]
					&& curr[idx_local] <= next[idx_local] && curr[idx_local] <= next[idx_local + 1]
					&& curr[idx_local] <= next[idx_local - 1] && curr[idx_local] <= next[idx_local + 18]
					&& curr[idx_local] <= next[idx_local - 18] && curr[idx_local] <= next[idx_local + 18 + 1]
					&& curr[idx_local] <= next[idx_local + 18 - 1] && curr[idx_local] <= next[idx_local - 18 + 1]
					&& curr[idx_local] <= next[idx_local - 18 - 1]
					&& curr[idx_local] <= prev[idx_local] && curr[idx_local] <= prev[idx_local + 1]
					&& curr[idx_local] <= prev[idx_local - 1] && curr[idx_local] <= prev[idx_local + 18]
					&& curr[idx_local] <= prev[idx_local - 18] && curr[idx_local] <= prev[idx_local + 18 + 1]
					&& curr[idx_local] <= prev[idx_local + 18 - 1] && curr[idx_local] <= prev[idx_local - 18 + 1]
					&& curr[idx_local] <= prev[idx_local - 18 - 1])
				{
                    idx_atomic = atomic_add(&t_idx[0], 1); // get the private index for ouput
				}
			}  //abs(val)>threshold
		}  // gj>=img_border && gj<cols-img_border
	}  //gi>=img_border && gi<rows-img_border
    
    if(idx_atomic>=0){ // only for those who find a local extrema, and get the unique index
        kpts_list[idx_atomic*4+0] = gj;
        kpts_list[idx_atomic*4+1] = gk;
        kpts_list[idx_atomic*4+2] = Octave;
        kpts_list[idx_atomic*4+3] = gi+1;
    }
}

// optimized kernel
// used to eliminate the weak points, and calculate the exact numbers of keypoints
__kernel void adjustLocalExtrema(__global int* dogpyr, // input, dog pyramid
                                 __global int* t_idx, // used to trace the output list length
                                 __global float* kpts_out, // output list of keypoints
                                 __global float* keypoints, // input list of local extrema
                                 const unsigned int rows, 
                                 const unsigned int cols, 
                                 const unsigned int nOctaveLayers, 
                                 const float contrastThreshold, // contrast threshold, used to eliminate low contrast point
                                 const float edgeThreshold, // edge threshold, used to eliminate edge points
                                 const float sigma, // parameter for gaussian blur
                                 const unsigned int max_interp_step, // used to position a keypoint
                                 const unsigned int img_border,
                                 const unsigned int global_size) 
{
	unsigned int gi = get_global_id(0);
    
    if(gi >= global_size)
        return;
    // init global index tracer
    if(gi == 0)
        t_idx[0] = 0;
    
    // coefficients used for following step
	const float deriv_scale = 1.0 / 255.0*0.5;
	const float second_deriv_scale = 1.0 / 255.0;
	const float cross_deriv_scale = 1.0 / 255.0*0.25;
    
    // information of the local extrema
	int r = (int)keypoints[gi * 4 + 0];
	int c = (int)keypoints[gi * 4 + 1];
	int layer = (int)keypoints[gi * 4 + 3];
	int octv = (int)keypoints[gi * 4 + 2];
    
    // used to represent how strong the contrast point is
	float contr = -1.f;
    
	int i = 0;
    
    // used to store the solution of least-squares solution to a linear matrix equation
    float X[3];
    
    // index of the corresponding image pixels
    int idx = layer*rows*cols + r*cols + c;
    
	for (; i < max_interp_step; i++) {
		// solve the least-squares solution to a linear matrix equation dD = -H * X
        // see https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics) for more information
		
        float G[9];
        float E[3];
        {   
            float dD[3];
            dD[0] = (dogpyr[idx + 1] - dogpyr[idx - 1]) * deriv_scale;
            dD[1] = (dogpyr[idx + cols] - dogpyr[idx - cols]) * deriv_scale;
            dD[2] = (dogpyr[idx + cols*rows] - dogpyr[idx - cols*rows]) * deriv_scale;
            
            float H[9];
            {
                float dxx = (dogpyr[idx + 1] + dogpyr[idx - 1] - dogpyr[idx] * 2)*second_deriv_scale;
                float dyy = (dogpyr[idx + cols] + dogpyr[idx - cols] - dogpyr[idx] * 2)*second_deriv_scale;
                float dss = (dogpyr[idx + cols*rows] + dogpyr[idx - cols*rows] - dogpyr[idx] * 2)*second_deriv_scale;
                float dxy = (dogpyr[idx + cols + 1] + dogpyr[idx - cols - 1] - dogpyr[idx + cols - 1] - dogpyr[idx - cols + 1])*cross_deriv_scale;
                float dxs = (dogpyr[idx + cols*rows + 1] - dogpyr[idx + cols*rows - 1] - dogpyr[idx - cols*rows + 1] + dogpyr[idx - cols*rows - 1])*cross_deriv_scale;
                float dys = (dogpyr[idx + cols*rows + cols] - dogpyr[idx + cols*rows - cols] - dogpyr[idx - cols*rows + cols] + dogpyr[idx - cols*rows - cols])*cross_deriv_scale;
                H[0] = dxx;
                H[1] = dxy;
                H[2] = dxs;
                H[3] = dxy;
                H[4] = dyy;
                H[5] = dys;
                H[6] = dxs;
                H[7] = dys;
                H[8] = dss;
            }

            G[0] = H[0]*H[0] + H[3]*H[3] + H[6]*H[6];
            G[1] = H[1]*H[0] + H[4]*H[3] + H[7]*H[6];
            G[2] = H[2]*H[0] + H[5]*H[3] + H[8]*H[6];
            G[3] = H[0]*H[1] + H[3]*H[4] + H[6]*H[7];
            G[4] = H[1]*H[1] + H[4]*H[4] + H[7]*H[7];
            G[5] = H[2]*H[1] + H[5]*H[4] + H[8]*H[7];
            G[6] = H[0]*H[2] + H[3]*H[5] + H[6]*H[8];
            G[7] = H[1]*H[2] + H[4]*H[5] + H[7]*H[8];
            G[8] = H[2]*H[2] + H[5]*H[5] + H[8]*H[8];
            
            E[0] = H[0]*dD[0] + H[3]*dD[1] + H[6]*dD[2];
            E[1] = H[1]*dD[0] + H[4]*dD[1] + H[7]*dD[2];
            E[2] = H[2]*dD[0] + H[5]*dD[1] + H[8]*dD[2];
        }
        
        float G_inverse[9];
		float G_det = G[0] * (G[4] * G[8] - G[5] * G[7]) - G[1] * (G[3] * G[8] - G[5] * G[6]) + G[2] * (G[3] * G[7] - G[4] * G[6]);
        
        if(G_det == 0)
            return;
        
		G_inverse[0] = (G[4] * G[8] - G[7] * G[5]) / G_det;
		G_inverse[1] = (G[2] * G[7] - G[1] * G[8]) / G_det;
		G_inverse[2] = (G[1] * G[5] - G[2] * G[4]) / G_det;
		G_inverse[3] = (G[5] * G[6] - G[3] * G[8]) / G_det;
		G_inverse[4] = (G[0] * G[8] - G[2] * G[6]) / G_det;
		G_inverse[5] = (G[3] * G[2] - G[0] * G[5]) / G_det;
		G_inverse[6] = (G[3] * G[7] - G[6] * G[4]) / G_det;
		G_inverse[7] = (G[6] * G[1] - G[0] * G[7]) / G_det;
		G_inverse[8] = (G[0] * G[4] - G[3] * G[1]) / G_det;
        
        // get the results
		X[0] = -(G_inverse[0] * E[0] + G_inverse[1] * E[1] + G_inverse[2] * E[2]);
		X[1] = -(G_inverse[3] * E[0] + G_inverse[4] * E[1] + G_inverse[5] * E[2]);
		X[2] = -(G_inverse[6] * E[0] + G_inverse[7] * E[1] + G_inverse[8] * E[2]);

        // if small bias, the location is accurate enough, directly go to next step
		if (fabs(X[0]) < 0.5f && fabs(X[1]) < 0.5f && fabs(X[2]) < 0.5f)
			break;

        // eliminate weak point
		if (fabs(X[0]) > (float)(INT_MAX/3) || fabs(X[1]) > (float)(INT_MAX/3) || fabs(X[2]) > (float)(INT_MAX/3))
			return;
        
        // if not small bias, update the location of keypoint
		c += round(X[0]);
		r += round(X[1]);
		layer += round(X[2]);
        
        // out of boundary
		if (layer<1 || layer> nOctaveLayers || c < img_border || c >= cols - img_border || r < img_border || r >= rows - img_border)
			return;
	}
    
    // if still cannot find X < 0.5 after max_interp_step loops, discard the point
	if (i >= max_interp_step)  
		return;

    // eliminate weak point
	{
		float dD[3];
		dD[0] = (dogpyr[idx + 1] - dogpyr[idx - 1]) * deriv_scale;
        dD[1] = (dogpyr[idx + cols] - dogpyr[idx - cols]) * deriv_scale;
        dD[2] = (dogpyr[idx + cols*rows] - dogpyr[idx - cols*rows]) * deriv_scale;

		float t = dD[0] * X[0] + dD[1] * X[1] + dD[2] * X[2];
        
        if(t >= 3)
            return;
        
		contr = (float)dogpyr[idx] * (1.0 / 255.0) + t*0.5f;
        
        // low contrast point
		if (fabs(contr)*((float)nOctaveLayers) < contrastThreshold)  
			return;
        
		float dxx = (dogpyr[idx + 1] + dogpyr[idx - 1] - dogpyr[idx] * 2)*second_deriv_scale;
		float dyy = (dogpyr[idx + cols] + dogpyr[idx - cols] - dogpyr[idx] * 2)*second_deriv_scale;
		float dxy = (dogpyr[idx + cols + 1] + dogpyr[idx - cols - 1] - dogpyr[idx + cols - 1] - dogpyr[idx - cols + 1])*cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx*dyy - dxy*dxy;
        
        // edge point
		if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)  
			return;
	}
    
    float temp = (X[2]+0.5)*255;
    
    // get the unique index for the survived thread
    int idx_atomic = atomic_add(&t_idx[0], 1);
    
    kpts_out[idx_atomic*9+0] = (float)r; // position
    kpts_out[idx_atomic*9+1] = (float)c; // position
    kpts_out[idx_atomic*9+2] = (float)(octv + (layer<<8) + (( temp<0.5 ? (int)temp : (int)temp+1)<<16));  // octave
    kpts_out[idx_atomic*9+3] = (float)sigma* pow(2.0f, (float) (layer+X[2])/nOctaveLayers) * (1<<octv) *2;  // size
    kpts_out[idx_atomic*9+4] = (float)fabs(contr);  // response
    kpts_out[idx_atomic*9+5] = (float)layer; // layer
    kpts_out[idx_atomic*9+6] = (float)(1<<octv); // parameter for next step
    kpts_out[idx_atomic*9+7] = (float)X[0]; // parameter for next step
    kpts_out[idx_atomic*9+8] = (float)X[1]; // parameter for next step
}

// optimized kernel
// calculate the orientation of every keypoint
__kernel void calcOrientationHist(__global unsigned char* gpyr, // input, gaussian pyramid
                                  __global int* t_idx, // used to trace the output list length
                                  __global float* kpts_out, // output list of keypoints
                                  __global float* keypoints, // input information about keypoints
                                  const unsigned int rows, 
                                  const unsigned int cols, 
                                  const int firstoctave, // = -1
                                  const unsigned int n,  // = 36
                                  const unsigned int ori_radius, // = 4.5
                                  const unsigned int ori_sig_fctr, // = 1.5
                                  const float ori_peak_ratio,
                                  const unsigned int global_size) // =0.8
{
	unsigned int gi = get_global_id(0);
    if(gi >= global_size)
        return;
    
    // init global index tracer
    if(gi == 0)
        t_idx[0] = 0;
    
    float scl_octv = keypoints[gi*9+3] * 0.5f / (float)(1 << (int)keypoints[gi*9+2]);
    
    // radius of neighbors
    unsigned int radius = ori_radius * scl_octv;
    float sigma = ori_sig_fctr * scl_octv;
    int layer = keypoints[gi*9 + 5];
    
    // the number of neighbors
    int len = (radius * 2 + 1)*(radius * 2 + 1);
    
    // used to store histogram of neighbors
    float temphist[40];
    float hist[36];
    
    float maxval;
    int k = 0;
    
    {
        // used to store information of neighbors
        // seems these variables will be located in global memory automatically
        // the reason of 1681 is to avoid information loss due to large 'len'
        float Mag[1681];  // magnitude
        int Ori[1681];  // orientation
        float W[1681];  // weight
        
        for (int i = 0; i < n; i++)
            temphist[i + 2] = 0.0;
        
        for (int i = 0; i <= 2*radius; i++) {
            int y = (int)keypoints[gi * 9 + 1] + i - radius;
            
            // out of range
            if (y <= 0 || y >= rows - 1)
                continue;
            
            for (int j = 0; j <= 2*radius; j++) {
                int x = (int)keypoints[gi * 9 + 0] + j - radius;
                
                // out of range
                if (x <= 0 || x >= cols - 1)
                    continue;
                
                // avoid overflow
                if(i*(2*radius+1)+j > 1681)
                    break;
                
                float dx = gpyr[layer*rows*cols + y*cols + (x + 1)] - gpyr[layer*rows*cols + y*cols + (x - 1)];
                float dy = gpyr[layer*rows*cols + (y - 1)*cols + x] - gpyr[layer*rows*cols + (y + 1)*cols + x];
                W[i*(2*radius+1)+j] = exp((float)(i*i + j*j)*-1.0 / (2.0*sigma*sigma));
                Mag[i*(2*radius+1)+j] = sqrt(dx*dx + dy*dy);
                Ori[i*(2*radius+1)+j] = atan2(dy, dx) * 180 / 3.141592653589793f;  // -180 ~ +180
                k += 1;
            }
        }
        
        // avoid mistake
        len = k;
        
        float temp[36];
        for (k = 0; k < len; k++) {
            Ori[k] = Ori[k]<0 ? Ori[k] + 360 : Ori[k];// 0 ~ +180 , -180+360 ~ 0+360
            int bin = round((float)(n / 360.0) * Ori[k]);
            if(bin < 0 || bin >= n)
                continue;
            
            temp[bin] += W[k] * Mag[k];
        }
        
        // This is a redundant step, but if I fixed this, there will be an out-of-resource problem, can't figure why
        for(k=0; k<n;k++)
            temphist[k+2] = temp[k];
    }
    
	// smooth the histogram
	temphist[1] = temphist[n + 1];
	temphist[0] = temphist[n];
	temphist[n + 2] = temphist[2];
	temphist[n + 3] = temphist[3];
	for (int i = 0; i < n; i++) {
		hist[i] = (temphist[i] + temphist[i + 4])*(1.f / 16.f) + (temphist[i + 1] + temphist[i + 3])*(4.f / 16.f) + temphist[i + 2] * (6.f / 16.f);
	}
    
    // find the max value
	maxval = hist[0];
	for (int i = 1; i < n; i++){
		maxval = maxval < hist[i] ? hist[i] : maxval;
    }
    
    // to allow multi orientation
    maxval *= ori_peak_ratio;
    
    float scale = 1.f / (float)(1 << -firstoctave);
    
    // find all the main orientation
    for (k = 0; k < n; k++ ){
        int left = k > 0 ? k - 1 : n - 1;
        int righ = k < n-1 ? k + 1 : 0;
        
        if (hist[k] > hist[left] && hist[k] > hist[righ] && hist[k] > maxval){
            float bin = k + 0.5f * (hist[left] - hist[righ]) / (hist[left] + hist[righ] - 2*hist[k]);
            float angle = 360.f - (float)(360.f/n) * bin;
            // avoid trash number
            if(angle < 0 || angle >= 360)
                continue;
            if (fabs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;
            
            int idx_atomic = atomic_add(&t_idx[0], 1);
            int octave = keypoints[gi*9+2];
            kpts_out[idx_atomic*6+0] = (float)(keypoints[gi*9+0] + keypoints[gi*9+7]) * keypoints[gi*9+6] * scale;  // pt.x
            kpts_out[idx_atomic*6+1] = (float)(keypoints[gi*9+1] + keypoints[gi*9+8]) * keypoints[gi*9+6] * scale;  // pt.y
            kpts_out[idx_atomic*6+2] = (float)((octave & ~255) | ((octave + firstoctave) & 255));  // octave
            kpts_out[idx_atomic*6+3] = (float)keypoints[gi*9+3];  // size
            kpts_out[idx_atomic*6+4] = (float)keypoints[gi*9+4];  // response
            kpts_out[idx_atomic*6+5] = (float)angle;  // orientation
        }
    }
}

