local cunn = require 'cunn'
require 'BatchIterator'
require 'Localizer'

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects,
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0, 0, s[3], s[2]))
  local idx = { {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end
local function stateHarvest(net)
  local name = "nn.ReLU"
  for k,v in pairs(net:findModules(name)) do
    print(v.val)
  end
end

function create_objective(model, weights, gradient, batch_iterator, stats,confusion)
  local cfg = model.cfg
  local pnet = model.pnet
  stateHarvest(pnet)
  local bgclass = cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors
  local localizer = Localizer.new(pnet.outnode.children[5])

  local softmax = nn.CrossEntropyCriterion():cuda()
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()

  local function cleanAnchors(examples, outputs)
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      local fmSize = outputs[anchor.layer]:size()
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end
    end
  end

  local function lossAndGradient(w)
    if w ~= weights then
      weights:copy(w)
    end
    gradient:zero()
    -- statistics for proposal stage
    local cls_loss = 0
    --local reg_loss = 0
    local cls_count = 0
    --local reg_count = 0
    local delta_outputs = {}


    -- enable dropouts
    pnet:training()

    local batch = batch_iterator:nextTraining()
    --print('batch')
    --print(batch)
    local target = torch.Tensor(1,2):zero()
    target[{1,1}] = 1
    target[{1,2}] = 0
    local clsOutput = torch.Tensor(1,2):zero()
    clsOutput[{1,1}] = 1
    clsOutput[{1,2}] = 0
    for i,x in ipairs(batch) do
      local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
      local p = x.positive        -- get positive and negative anchors examples
      local n = x.negative

      -- run forward convolution
      local outputs = pnet:forward(img)

      -- ensure all example anchors lie withing existing feature planes
      cleanAnchors(p, outputs)
      cleanAnchors(n, outputs)

      -- clear delta values for each new image
      for i,out in ipairs(outputs) do
        if not delta_outputs[i] then
          delta_outputs[i] = torch.FloatTensor():cuda()
        end
        delta_outputs[i]:resizeAs(out)
        delta_outputs[i]:zero()
      end

      local roi_pool_state = {}
      local input_size = img:size()

      target[{1,1}] = 1
      target[{1,2}] = 0
      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = anchor.layer

        local out = outputs[l]
        local delta_out = delta_outputs[l]

        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        -- classification
        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
        local dc = softmax:backward(v[{{1, 2}}], 1)
        d[{{1,2}}]:add(dc)
        d[{{3,6}}]:zero()
        clsOutput[{1,1}]=v[1]
        clsOutput[{1,2}]=v[2]
        confusion:batchAdd(clsOutput, target)
      end

      target[{1,1}] = 0
      target[{1,2}] = 1
      -- process negative
      for i,x in ipairs(n) do
        local anchor = x[1]
        local l = anchor.layer
        local out = outputs[l]
        local delta_out = delta_outputs[l]
        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
        local dc = softmax:backward(v[{{1, 2}}], 2)
        d[{{1,2}}]:add(dc)
        d[{{3,6}}]:zero()
        clsOutput[{1,1}]=v[1]
        clsOutput[{1,2}]=v[2]

        confusion:batchAdd(clsOutput, target)
      end

      -- backward pass of proposal network
      pnet:backward(img, delta_outputs)

      print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
      cls_count = cls_count + #p + #n
    end

    local pcls = cls_loss / cls_count     -- proposal classification (bg/fg)


    print(string.format('prop: cls: %f (%d)', pcls, cls_count))
    confusion:updateValids()
    local train_acc = confusion.totalValid * 100
    table.insert(stats.pcls, pcls)
    table.insert(stats.train_acc, train_acc)


    local loss = pcls --+ preg
    return loss, gradient
  end

  return lossAndGradient
end
