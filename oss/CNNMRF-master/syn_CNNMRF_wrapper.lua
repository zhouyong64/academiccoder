require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'loadcaffe'

paths.dofile('mylib/myoptimizer.lua')
paths.dofile('mylib/tv.lua')
paths.dofile('mylib/mrf.lua')
paths.dofile('mylib/helper.lua')

torch.setdefaulttensortype('torch.FloatTensor') -- float as default tensor type

local function main(params)
  os.execute('mkdir data/result/')
  os.execute('mkdir data/result/freesyn/')
  os.execute('mkdir data/result/freesyn/MRF/')
  os.execute(string.format('mkdir %s', params.output_folder))

  local net = nn.Sequential()
  local i_net_layer = 0
  local num_calls = 0
  local next_mrf_idx = 1
  local mrf_losses = {}
  local mrf_layers = {}
  local i_mrf_layer = 0
  local input_image
  local output_image
  local cur_res
  local mrf_layers_pretrained = params.mrf_layers

  -----------------------------------------------------------------------------------
  -- read images
  -----------------------------------------------------------------------------------
  local source_image = image.load(string.format('data/content/%s.jpg', params.content_name), 3)
  local target_image = image.load(string.format('data/style/%s.jpg', params.style_name), 3)

  source_image = image.scale(source_image, params.max_size, 'bilinear')
  target_image = image.scale(target_image, math.floor(params.max_size / params.scaler), 'bilinear')

  local render_height = source_image:size()[2]
  local render_width = source_image:size()[3]
  local source_image_caffe = preprocess(source_image):float()
  local target_image_caffe = preprocess(target_image):float()

  local pyramid_source_image_caffe = {}
  for i_res = 1, params.num_res do
    pyramid_source_image_caffe[i_res] = image.scale(source_image_caffe, 
      math.ceil(source_image:size()[3] * math.pow(0.5, params.num_res - i_res)), 
         math.ceil(source_image:size()[2] * math.pow(0.5, params.num_res - i_res)), 'bilinear')
  end

  local pyramid_target_image_caffe = {}
  for i_res = 1, params.num_res do
    pyramid_target_image_caffe[i_res] = image.scale(target_image_caffe, 
      math.ceil(target_image:size()[3] * math.pow(0.5, params.num_res - i_res)), 
         math.ceil(target_image:size()[2] * math.pow(0.5, params.num_res - i_res)), 'bilinear')
  end

  -- --------------------------------------------------------------------------------------------------------
  -- -- local function for adding a mrf layer, with image rotation andn scaling
  -- --------------------------------------------------------------------------------------------------------
  local function add_mrf()
      local mrf_module = nn.MRFMM()
      i_mrf_layer = i_mrf_layer + 1
      i_net_layer = i_net_layer + 1
      next_mrf_idx = next_mrf_idx + 1
      if params.gpu >= 0 then
        if params.backend == 'cudnn' then
          mrf_module:cuda()
        else
          mrf_module:cl()
        end
      end
      net:add(mrf_module)
      table.insert(mrf_losses, mrf_module)
      table.insert(mrf_layers, i_mrf_layer, i_net_layer)
      return true
  end

  local function build_mrf(id_mrf)
    --------------------------------------------------------
    -- deal with target
    --------------------------------------------------------
    local target_images_caffe = {}
    for i_r = -params.target_num_rotation, params.target_num_rotation do
      local alpha = params.target_step_rotation * i_r
      local min_x, min_y, max_x, max_y = computeBB(pyramid_target_image_caffe[cur_res]:size()[3], 
                                           pyramid_target_image_caffe[cur_res]:size()[2], alpha)
      local target_image_rt_caffe = image.rotate(pyramid_target_image_caffe[cur_res], alpha, 'bilinear')
      target_image_rt_caffe = 
           target_image_rt_caffe[{{1, target_image_rt_caffe:size()[1]}, {min_y, max_y}, {min_x, max_x}}]

      for i_s = -params.target_num_scale, params.target_num_scale do
        local max_sz = math.floor(math.max(target_image_rt_caffe:size()[2], target_image_rt_caffe:size()[3]) * 
                                                               torch.pow(params.target_step_scale, i_s))
        local target_image_rt_s_caffe = image.scale(target_image_rt_caffe, max_sz, 'bilinear')
        if params.gpu >= 0 then
          if params.backend == 'cudnn' then
            target_image_rt_s_caffe = target_image_rt_s_caffe:cuda()
          else
            target_image_rt_s_caffe = target_image_rt_s_caffe:cl()
          end
        end
        table.insert(target_images_caffe, target_image_rt_s_caffe)
      end
    end

    -- compute the coordinates on the pixel layer
    local target_x
    local target_y
    local target_x_per_image = {}
    local target_y_per_image = {}
    local target_imageid
    -- print('*****************************************************')
    -- print(string.format('build target mrf'));
    -- print('*****************************************************')
    for i_image = 1, #target_images_caffe do
      -- print(string.format('image %d, ', i_image))
      net:forward(target_images_caffe[i_image])
      local target_feature_map = net:get(mrf_layers[id_mrf] - 1).output:float()

      if params.mrf_patch_size[id_mrf] > target_feature_map:size()[2] or 
        params.mrf_patch_size[id_mrf] > target_feature_map:size()[3] then
        print('target_images is not big enough for patch')
        print('target_images size: ')
        print(target_feature_map:size())
        print('patch size: ')
        print(params.mrf_patch_size[id_mrf])
        do return end
      end
      local target_x_, target_y_ = drill_computeMRFfull(target_feature_map,  
             params.mrf_patch_size[id_mrf], params.target_sample_stride[id_mrf], -1)


      local x = torch.Tensor(target_x_:nElement() * target_y_:nElement())
      local y = torch.Tensor(target_x_:nElement() * target_y_:nElement())
      local target_imageid_ = torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(i_image)
      local count = 1
      for i_row = 1, target_y_:nElement() do
        for i_col = 1, target_x_:nElement() do
          x[count] = target_x_[i_col]
          y[count] = target_y_[i_row]
          count = count + 1
        end
      end
      if i_image == 1 then
        target_x = x:clone()
        target_y = y:clone()
        target_imageid = target_imageid_:clone()
      else
        target_x = torch.cat(target_x, x, 1)
        target_y = torch.cat(target_y, y, 1)
        target_imageid = torch.cat(target_imageid, target_imageid_, 1)
      end
      table.insert(target_x_per_image, x)
      table.insert(target_y_per_image, y)
    end -- end for i_image = 1, #target_images do

    -- print('*****************************************************')
    -- print(string.format('collect mrf'));
    -- print('*****************************************************')

    local num_channel_mrf = net:get(mrf_layers[id_mrf] - 1).output:size()[1]
    local target_mrf = torch.Tensor(target_x:nElement(), 
          num_channel_mrf * params.mrf_patch_size[id_mrf] * params.mrf_patch_size[id_mrf])
    local tensor_target_mrf = torch.Tensor(target_x:nElement(), num_channel_mrf, 
                             params.mrf_patch_size[id_mrf], params.mrf_patch_size[id_mrf])
    local count_mrf = 1
    for i_image = 1, #target_images_caffe do
      -- print(string.format('image %d, ', i_image));
      net:forward(target_images_caffe[i_image])
      -- sample mrf on mrf_layers
      local tensor_target_mrf_, target_mrf_ = sampleMRFAndTensorfromLocation2(target_x_per_image[i_image], 
        target_y_per_image[i_image], net:get(mrf_layers[id_mrf] - 1).output:float(), params.mrf_patch_size[id_mrf])
      target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, target_mrf:size()[2]}}] = target_mrf_:clone()
      tensor_target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, tensor_target_mrf:size()[2]}, 
                      {1, tensor_target_mrf:size()[3]}, {1, tensor_target_mrf:size()[4]}}] = tensor_target_mrf_:clone()
      count_mrf = count_mrf + target_mrf_:size()[1]
      tensor_target_mrf_ = nil
      target_mrf_ = nil
      collectgarbage()
    end --for i_image = 1, #target_images do
    local target_mrfnorm = 
                torch.sqrt(torch.sum(torch.cmul(target_mrf, target_mrf), 2)):resize(target_mrf:size()[1], 1, 1)

    --------------------------------------------------------
    -- process source
    --------------------------------------------------------
    -- print('*****************************************************')
    -- print(string.format('process source image'));
    -- print('*****************************************************')
    if params.gpu >= 0 then
      if params.backend == 'cudnn' then
        net:forward(pyramid_source_image_caffe[cur_res]:cuda())
      else
        net:forward(pyramid_source_image_caffe[cur_res]:cl())
      end
    else
      net:forward(pyramid_source_image_caffe[cur_res])
    end
    local source_feature_map = net:get(mrf_layers[id_mrf] - 1).output:float()
    if params.mrf_patch_size[id_mrf] > source_feature_map:size()[2] or 
           params.mrf_patch_size[id_mrf] > source_feature_map:size()[3] then
      print('source_image_caffe is not big enough for patch')
      print('source_image_caffe size: ')
      print(source_feature_map:size())
      print('patch size: ')
      print(params.mrf_patch_size[id_mrf])
      do return end
    end
    local source_xgrid, source_ygrid = drill_computeMRFfull(source_feature_map:float(), 
               params.mrf_patch_size[id_mrf], params.source_sample_stride[id_mrf], -1)
    local source_x = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())
    local source_y = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())
    local count = 1
    for i_row = 1, source_ygrid:nElement() do
      for i_col = 1, source_xgrid:nElement() do
        source_x[count] = source_xgrid[i_col]
        source_y[count] = source_ygrid[i_row]
        count = count + 1
      end
    end
    -- local tensor_target_mrfnorm = torch.repeatTensor(target_mrfnorm:float(), 1, 
        --net:get(mrf_layers[id_mrf] - 1).output:size()[2] - (params.mrf_patch_size[id_mrf] - 1), 
        --net:get(mrf_layers[id_mrf] - 1).output:size()[3] - (params.mrf_patch_size[id_mrf] - 1))

    -- print('*****************************************************')
    -- print(string.format('call layer implemetation'));
    -- print('*****************************************************')
    local nInputPlane = target_mrf:size()[2] / (params.mrf_patch_size[id_mrf] * params.mrf_patch_size[id_mrf])
    local nOutputPlane = target_mrf:size()[1]
    local kW = params.mrf_patch_size[id_mrf]
    local kH = params.mrf_patch_size[id_mrf]
    local dW = 1
    local dH = 1
    local input_size = source_feature_map:size()

    local source_xgrid_, source_ygrid_ = 
           drill_computeMRFfull(source_feature_map:float(), params.mrf_patch_size[id_mrf], 1, -1)
    local response_size = torch.LongStorage(3)
    response_size[1] = nOutputPlane
    response_size[2] = source_ygrid_:nElement()
    response_size[3] = source_xgrid_:nElement()
    net:get(mrf_layers[id_mrf]):implement(params.mode, target_mrf, tensor_target_mrf, target_mrfnorm, 
          source_x, source_y, input_size, response_size, nInputPlane, nOutputPlane, kW, kH, 1, 1, 
          params.mrf_confidence_threshold[id_mrf], params.mrf_weight[id_mrf], params.gpu_chunck_size_1, 
          params.gpu_chunck_size_2, params.backend, params.gpu)
    target_mrf = nil
    tensor_target_mrf = nil
    source_feature_map = nil
    collectgarbage()
  end

  --------------------------------------------------------------------------------------------------------
  -- local function for printing inter-mediate result
  --------------------------------------------------------------------------------------------------------
  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
        print(string.format('Iteration %d, %d', t, params.num_iter[cur_res]))
     end
  end

  --------------------------------------------------------------------------------------------------------
  -- local function for saving inter-mediate result
  --------------------------------------------------------------------------------------------------------
  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iter
    if should_save then
    local disp = deprocess(input_image:float())
    disp = image.minmax{tensor=disp, min=0, max=1}
    disp = image.scale(disp, render_width, render_height, 'bilinear')
    local filename = string.format('%s/res_%d_%d.jpg', params.output_folder, cur_res, t)
    image.save(filename, disp)
    end
  end

  --------------------------------------------------------------------------------------------------------
  -- local function for computing energy
  --------------------------------------------------------------------------------------------------------
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:backward(x, dy)
    local loss = 0
    collectgarbage()

    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -------------------------------------------------------------------------------
  -- initialize network
  -------------------------------------------------------------------------------
  if params.gpu >= 0 then
    if params.backend == 'cudnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'cltorch'
      require 'clnn'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
  end

  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then
    loadcaffe_backend = 'nn'
  end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend == 'cudnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  print('cnn succesfully loaded')

  for i_res = 1, params.num_res do
    local timer = torch.Timer()

    cur_res = i_res
    num_calls = 0
    local optim_state = {
      maxIter = params.num_iter[i_res],
      nCorrection = params.nCorrection,
      verbose=true,
      tolX = 0,
      tolFun = 0,
    }

    -- initialize image and target
    if i_res == 1 then

      if params.ini_method == 'random' then
        input_image = torch.randn(pyramid_source_image_caffe[i_res]:size()):float():mul(0.001)
      elseif params.ini_method == 'image' then
        input_image = pyramid_source_image_caffe[i_res]:clone():float()
      else
        error('Invalid init type')
      end
      if params.gpu >= 0 then
        if params.backend == 'cudnn' then
          input_image = input_image:cuda()
        else
          input_image = input_image:cl()
        end
      end
      -----------------------------------------------------
      -- add a tv layer
      -----------------------------------------------------
      if params.tv_weight > 0 then
        local tv_mod = nn.TVLoss(params.tv_weight):float()
        if params.gpu >= 0 then
          if params.backend == 'cudnn' then
            tv_mod:cuda()
          else
            tv_mod:cl()
          end
        end
        i_net_layer = i_net_layer + 1
        net:add(tv_mod)
      end

      for i = 1, #cnn do
        if next_mrf_idx <= #mrf_layers_pretrained then
          local layer = cnn:get(i)

          i_net_layer = i_net_layer + 1
          net:add(layer)

          -- -- add mrfstatsyn layer
          if i == mrf_layers_pretrained[next_mrf_idx] then
            if add_mrf() == false then
              print('build network failed: adding mrf layer failed')
              do return end
            end
          end

        end
      end -- for i = 1, #cnn do

      cnn = nil
      collectgarbage()

      print(net)


      print('mrf_layers: ')
      for i = 1, #mrf_layers do
        print(mrf_layers[i])
      end

      print('network has been built.')
    else
      input_image = image.scale(input_image:float(), pyramid_source_image_caffe[i_res]:size()[3], 
                                pyramid_source_image_caffe[i_res]:size()[2], 'bilinear'):clone()
      if params.gpu >= 0 then
        if params.backend == 'cudnn' then
          input_image = input_image:cuda()
        else
          input_image = input_image:cl()
        end
      end
    end

    print('*****************************************************')
    print(string.format('Synthesis started at resolution ', cur_res))
    print('*****************************************************')

    print('Implementing mrf layers ...')
    for i = 1, #mrf_layers do
      if build_mrf(i) == false then
        print('build_mrf failed')
        do return end
      end
    end

    local mask = torch.Tensor(input_image:size()):fill(1)
    if params.gpu >= 0 then
      if params.backend == 'cudnn' then
        mask = mask:cuda()
      else
        mask = mask:cl()
      end
    end

    y = net:forward(input_image)
    dy = input_image.new(#y):zero()

    -- do optimizatoin
    local x, losses = mylbfgs(feval, input_image, optim_state, nil, mask)

    local t = timer:time().real
    print(string.format('Synthesis finished at resolution %d, %f seconds', cur_res, t))
  end

  net = nil
  source_image = nil
  target_image = nil
  pyramid_source_image_caffe = nil
  pyramid_target_image_caffe = nil
  input_image = nil
  output_image = nil
  mrf_losses = nil
  mrf_layers = nil
  optim_state = nil
  collectgarbage()
  collectgarbage()

end -- end of main


local function run_test(content_name, style_name, ini_method, max_size, scaler, num_res, num_iter, 
   mrf_layers, mrf_weight, mrf_patch_size, mrf_num_rotation, mrf_num_scale, mrf_sample_stride, 
  mrf_synthesis_stride, mrf_confidence_threshold, tv_weight, mode,  
  gpu_chunck_size_1, gpu_chunck_size_2, backend)

  -- local clock = os.clock
  -- function sleep(n)  -- seconds
  --   local t0 = clock()
  --   while clock() - t0 <= n do end
  -- end

  local timer_TEST = torch.Timer()

  local flag_state = 1

  local params = {}

  -- externally set paramters
  params.content_name = content_name
  params.style_name = style_name
  --print(backend)
  params.ini_method = ini_method  
  params.max_size = max_size or 384
  params.scaler = scaler or 2
  params.num_res = num_res or 3
  params.num_iter = num_iter or {100, 100, 100}
  params.mrf_layers = mrf_layers or {12, 21}
  params.mrf_weight = mrf_weight or {1e-4, 1e-4}
  params.mrf_patch_size = mrf_patch_size or {3, 3}
  params.target_num_rotation = mrf_num_rotation or 0
  params.target_num_scale = mrf_num_scale or 0
  params.target_sample_stride = mrf_sample_stride or {2, 2}
  params.source_sample_stride = mrf_synthesis_stride or {2, 2}
  params.mrf_confidence_threshold = mrf_confidence_threshold or {0, 0}
  params.tv_weight = tv_weight or 1e-3

  params.mode = mode or 'speed'
  params.gpu_chunck_size_1 = gpu_chunck_size_1 or 256
  params.gpu_chunck_size_2 = gpu_chunck_size_2 or 16
  params.backend = backend or 'cudnn'

  -- fixed parameters
  params.target_step_rotation = math.pi/24
  params.target_step_scale = 1.05

  params.proto_file = 'data/models/VGG_ILSVRC_19_layers_deploy.prototxt'
  params.model_file = 'data/models/VGG_ILSVRC_19_layers.caffemodel'
  params.gpu = 0
  params.nCorrection = 25
  params.print_iter = 10
  params.save_iter = 10
  params.gpu_chunck_size_1 = 32
  params.gpu_chunck_size_2 = 2

  params.output_folder = string.format('data/result/freesyn/MRF/%s_TO_%s', 
                           params.content_name, params.style_name)

  main(params)

  local t_test = timer_TEST:time().real
  print(string.format('Total time: %f seconds', t_test))
  -- sleep(1)
  return flag_state
end

return {
  run_test = run_test,
  main = main
}
