document.addEventListener('DOMContentLoaded', function() {
    const fineTuningMethodSelect = document.getElementById('fine-tuning-method');
    const loraParamsSection = document.getElementById('lora-params-section');
    const precisionSelect = document.getElementById('precision');
    const hardwareSelect = document.getElementById('hardware');
    const allHardwareOptions = Array.from(hardwareSelect.options);

    fineTuningMethodSelect.addEventListener('change', function() {
        if (fineTuningMethodSelect.value === 'lora') {
            loraParamsSection.style.display = 'block';
        } else {
            loraParamsSection.style.display = 'none';
        }
    });

    precisionSelect.addEventListener('change', function() {
        const selectedPrecision = precisionSelect.value;
        hardwareSelect.innerHTML = '';

        if (selectedPrecision === 'fp8') {
            hardwareSelect.add(allHardwareOptions.find(option => option.value === 'nvidia_h20').cloneNode(true));
            hardwareSelect.add(allHardwareOptions.find(option => option.value === 'nvidia_h800').cloneNode(true));
        } else {
            allHardwareOptions.forEach(option => {
                hardwareSelect.add(option.cloneNode(true));
            });
        }
    });
});


document.getElementById('calculate-button').addEventListener('click', function() {
    const modelType = document.getElementById('model-type').value;
    const precision = document.getElementById('precision').value;
    const concurrency = parseInt(document.getElementById('concurrency').value);
    const contextLength = parseInt(document.getElementById('context-length').value);
    const framework = document.getElementById('framework').value;
    const fineTuningMethod = document.getElementById('fine-tuning-method').value;
    const loraTrainableParams = parseFloat(document.getElementById('lora-trainable-params').value) || 0;
    const hardware = document.getElementById('hardware').value;

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>计算结果:</h2>';

    const calculationResults = calculateRequirements(modelType, precision, concurrency, contextLength, framework, fineTuningMethod, loraTrainableParams, hardware);

    if (calculationResults) {
        let hardwareRecommendationHTML = '';
        if (calculationResults.hardware_recommendation && typeof calculationResults.hardware_recommendation === 'object') {
            hardwareRecommendationHTML += '<div class="result-item"><strong>推荐算力卡数量:</strong></div>';
            const hardwareName = Object.keys(calculationResults.hardware_recommendation)[0];
            const count = calculationResults.hardware_recommendation[hardwareName];
            hardwareRecommendationHTML += `<div class="result-item">  - ${getHardwareDisplayName(hardwareName)}: <strong>${count} 块</strong></div>`;
        }


        resultsDiv.innerHTML += `
            <div class="result-item"><strong>模型:</strong> ${getModelDisplayName(modelType)}</div>
            <div class="result-item"><strong>精度:</strong> ${getPrecisionDisplayName(precision)}</div>
            <div class="result-item"><strong>并发数:</strong> ${concurrency}</div>
            <div class="result-item"><strong>上下文长度:</strong> ${contextLength} Tokens</div>
            <div class="result-item"><strong>推理框架:</strong> ${getFrameworkDisplayName(framework)}</div>
            <div class="result-item"><strong>微调方法:</strong> ${getFineTuningMethodDisplayName(fineTuningMethod)}</div>
            <div class="result-item"><strong>算力卡:</strong> ${getHardwareDisplayName(hardware)}</div>
            ${fineTuningMethod === 'lora' ? `<div class="result-item"><strong>LoRA 可训练参数:</strong> ${loraTrainableParams} Billion</div>` : ''}
            <hr>
            <div class="result-item"><strong>预估显存需求:</strong></div>
            <div class="result-item">  - 模型权重: <strong>${calculationResults.model_weights_memory}</strong></div>
            <div class="result-item">  - KV Cache: <strong>${calculationResults.kv_cache_memory}</strong></div>
            <div class="result-item">  - 激活内存: <strong>${calculationResults.activation_memory}</strong></div>
            <div class="result-item">  - 碎片化 & 其他: <strong>${calculationResults.other_memory}</strong></div>
            <div class="result-item"><strong>总显存需求: </strong> <strong>${calculationResults.memory}</strong></div>
            ${hardwareRecommendationHTML}
            <div class="result-item"><strong>预估算力需求:</strong> ${calculationResults.compute}</div>
            <div class="result-item"><strong>预估算力机台数:</strong> <strong>${calculationResults.machine_count} 台</strong></div>
            <div class="result-item"><strong>部署建议:</strong> ${calculationResults.deployment_recommendation}</div>
            <div class="result-item"><strong>建议:</strong> ${calculationResults.recommendation}</div>
            <p class="result-item" style="font-size: smaller; color: gray;">* 显存和算力均为估算值，实际情况可能因多种因素而异。</p>
            <p class="result-item" style="font-size: smaller; color: gray;">* 推理显存估算公式：总内存 = 模型权重内存 + KV Cache 内存 + 激活内存 + 碎片化内存</p>
            ${fineTuningMethod === 'lora' ? `<p class="result-item" style="font-size: smaller; color: gray;">* LoRA 微调会增加少量模型权重内存。</p>` : ''}
            <p class="result-item" style="font-size: smaller; color: gray;">* 模型架构参数 (层数、隐藏维度等) 基于 DeepSeek 模型近似配置。</p>
            <p class="result-item" style="font-size: smaller; color: gray;">* 算力卡数量为满足显存需求的**最少估算**，实际部署可能需要更多卡以满足性能需求。</p>
            <p class="result-item" style="font-size: smaller; color: gray;">* 算力机台数假设 NVIDIA 和 华为昇腾 机器每台都包含 8 张算力卡。</p>
        `;
    } else {
        resultsDiv.innerHTML += '<p>无法估算，请检查输入参数。</p>';
    }
});


function calculateRequirements(modelType, precision, concurrency, contextLength, framework, fineTuningMethod = 'inference', loraTrainableParamsBillion = 0, hardware) {
    let estimatedMemoryGB = 0;
    let computeLoad = "中等";
    let recommendation = "请根据实际情况调整参数和框架选择。";
    let hardwareRecommendation = {};
    let model_weights_memory_gb = 0;
    let kv_cache_memory_gb = 0;
    let activation_memory_gb = 0;
    let other_memory_gb = 0;
    let machine_count = 0;
    let deployment_recommendation = "";

    // **直接使用常量定义模型大小 (GB) - 来自用户提供的数据**
    const modelSizesGB = {
        'r1_671b': { 'fp16': 1342, 'bf16': 1342, 'fp8': 671, 'int8': 671, 'int4': 335.5 },
        'r1_70b':  { 'fp16': 140,  'bf16': 140,  'fp8': 70,  'int8': 70,  'int4': 43 },
        'r1_32b':  { 'fp16': 64,   'bf16': 64,   'fp8': 32,  'int8': 32,  'int4': 20 },
        'r1_14b':  { 'fp16': 28,   'bf16': 28,   'fp8': 14,  'int8': 14,  'int4': 9 },
        'r1_8b':   { 'fp16': 16,   'bf16': 16,   'fp8': 8,   'int8': 8,   'int4': 4.9 },
        'r1_7b':   { 'fp16': 14,   'bf16': 14,   'fp8': 7,   'int8': 7,   'int4': 4.7 },
        'r1_1.5b': { 'fp16': 3,    'bf16': 3,    'fp8': 1.5, 'int8': 1.5, 'int4': 0.75 }
    };

    // **模型架构细节 (来自用户提供的数据)**
    const modelArchParams = {
        'r1_671b': { params: 671, layers: 61, hidden_dim: 7168, kv_heads: 128, head_dim: 128, kv_compress_dim: 512, moe: true },
        'r1_1.5b': { params: 1.5, layers: 28, hidden_dim: 2020, kv_heads: 3, head_dim: 673, kv_compress_dim: null, moe: false },
        'r1_7b':   { params: 7, layers: 34, hidden_dim: 4096, kv_heads: 32, head_dim: 128, kv_compress_dim: null, moe: false },
        'r1_8b':   { params: 8, layers: 32, hidden_dim: 4096, kv_heads: 32, head_dim: 128, kv_compress_dim: null, moe: false },
        'r1_14b':  { params: 14, layers: 69, hidden_dim: 4096, kv_heads: 32, head_dim: 128, kv_compress_dim: null, moe: false },
        'r1_32b':  { params: 32, layers: 64, hidden_dim: 6400, kv_heads: 8, head_dim: 800, kv_compress_dim: null, moe: false },
        'r1_70b':  { params: 70, layers: 80, hidden_dim: 8192, kv_heads: 64, head_dim: 128, kv_compress_dim: null, moe: false }
    };

    const model_config = modelArchParams[modelType];

    const n_dtype_bytes = {
        'fp8': 1,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    };
    const dtype_bytes = n_dtype_bytes[precision];


    const token_num = contextLength;


    // 1. 模型权重内存 (直接从常量读取)
    model_weights_memory_gb = modelSizesGB[modelType][precision];


    // 2. KV Cache 内存
    let kvCacheSizeBytes = 0;
    if (model_config.moe) { // MoE 模型 (R1 671B)
        kvCacheSizeBytes = concurrency * contextLength * model_config.layers * 2 * model_config.kv_compress_dim * dtype_bytes;
    } else { // 标准 Transformer 模型 (蒸馏模型)
        kvCacheSizeBytes = concurrency * model_config.layers * 2 * contextLength * model_config.kv_heads * model_config.head_dim * dtype_bytes;
    }
    kv_cache_memory_gb = kvCacheSizeBytes / (1024 * 1024 * 1024);


    // 3. 激活内存
    const activationSizeBytes = concurrency * contextLength * model_config.layers * model_config.hidden_dim * dtype_bytes;
    activation_memory_gb = activationSizeBytes / (1024 * 1024 * 1024);


    // 4. 总内存和碎片化内存
    estimatedMemoryGB = model_weights_memory_gb + kv_cache_memory_gb + activation_memory_gb;
    other_memory_gb = estimatedMemoryGB * 0.2; // 碎片化及其他开销 (20%)
    estimatedMemoryGB += other_memory_gb;


    const hardwareMemoryGB = {
        'nvidia_a10': 24,
        'nvidia_a100': 80,
        'nvidia_a100_40g': 40,
        'nvidia_a800': 80,
        'nvidia_h20': 96,
        'nvidia_h800': 80,
        'nvidia_l40s': 48,
        'nvidia_rtx4090': 24,
        'ascend910b': 56
    };

    hardwareRecommendation = calculateHardwareCount(estimatedMemoryGB, hardwareMemoryGB, hardware);

    const cardsPerMachine = 8;
    if (hardwareRecommendation && hardwareRecommendation[hardware]) {
        machine_count = Math.ceil(hardwareRecommendation[hardware] / cardsPerMachine);
    } else {
        machine_count = 0;
    }


    let deploymentFactorCompute = 1;
    if (model_config.params >= 70) { // 70B 及以上模型
        deployment_recommendation += " 建议采用多机多卡或模型并行等分布式部署策略。";
        computeLoad = adjustComputeLoad(computeLoad, 1.5);
    } else if (model_config.params >= 7) { // 7B-70B 模型
        deployment_recommendation += " 建议采用多卡并行或张量并行等方式以提高吞吐量。";
        computeLoad = adjustComputeLoad(computeLoad, 1.2);
    } else { // 小型模型 (7B 以下)
        deployment_recommendation += " 可以尝试单卡部署，或使用多卡并行以支持更高并发。";
    }


    let hardwareComputeFactor = 1;
    switch (hardware) {
        case 'ascend910b': hardwareComputeFactor = 0.8; computeLoad = adjustComputeLoad(computeLoad, 0.8); recommendation += " 昇腾910b 性能可能略低于同级别N卡。"; break;
        case 'nvidia_a10': hardwareComputeFactor = 0.6; computeLoad = adjustComputeLoad(computeLoad, 0.6); recommendation += " A10 性能相对较低，适合中小型模型。"; break;
        case 'nvidia_a100': hardwareComputeFactor = 1.2; computeLoad = adjustComputeLoad(computeLoad, 1.2); break;
        case 'nvidia_a100_40g': hardwareComputeFactor = 1.1; computeLoad = adjustComputeLoad(computeLoad, 1.1); recommendation += " A100-40G 性能略低于 A100-80G。"; break;
        case 'nvidia_a800': hardwareComputeFactor = 1.1; computeLoad = adjustComputeLoad(computeLoad, 1.1); break;
        case 'nvidia_h20': hardwareComputeFactor = 1.3; computeLoad = adjustComputeLoad(computeLoad, 1.3); break;
        case 'nvidia_h800': hardwareComputeFactor = 1.5; computeLoad = adjustComputeLoad(computeLoad, 1.5); computeLoad = "非常高"; recommendation = " H800/H20 是高性能卡，适合大型模型。"; break;
        case 'nvidia_rtx4090': hardwareComputeFactor = 0.9; computeLoad = adjustComputeLoad(computeLoad, 0.9); recommendation += " RTX 4090 消费级卡，性价比高，但显存可能受限。"; break;
        case 'nvidia_l40s': hardwareComputeFactor = 1.0; computeLoad = adjustComputeLoad(computeLoad, 1.0); break;
    }
    computeLoad = adjustComputeLoad(computeLoad, hardwareComputeFactor);


    if (framework === 'vllm') {
        computeLoad = adjustComputeLoad(computeLoad, 1.1);
        recommendation += " vLLM 框架通常能提供更高的推理吞吐量。";
        deployment_recommendation += " 推荐使用 vLLM 框架进行高性能推理部署。";
    } else if (framework === 'llama_cpp') {
        computeLoad = adjustComputeLoad(computeLoad, 0.9);
        recommendation += " llama.cpp 框架适用于 CPU/GPU 混合推理场景。";
        deployment_recommendation += " llama.cpp 适用于 CPU/GPU 混合推理，如果资源有限或需要CPU参与推理，可以考虑。";
    } else if (framework === 'mindspore') {
        recommendation += " MindSpore 是华为昇腾平台的推荐框架，性能可能更优。";
        deployment_recommendation += " 对于华为昇腾 910B 平台，强烈推荐使用 MindSpore 框架以获得最佳性能。";
    } else {
        deployment_recommendation += " 通用部署建议：根据模型大小和并发需求，选择合适的推理框架。";
    }


    return {
        memory: estimatedMemoryGB.toFixed(2) + " GB (估算值)",
        model_weights_memory: model_weights_memory_gb.toFixed(2) + " GB (估算值)",
        kv_cache_memory: kv_cache_memory_gb.toFixed(2) + " GB (估算值)",
        activation_memory: activation_memory_gb.toFixed(2) + " GB (估算值)",
        other_memory: other_memory_gb.toFixed(2) + " GB (估算值)",
        compute: computeLoad + " (估算值)",
        recommendation: recommendation,
        hardware_recommendation: hardwareRecommendation,
        machine_count: machine_count,
        deployment_recommendation: deployment_recommendation
    };
}


function calculateHardwareCount(estimatedMemoryGB, hardwareMemoryGB, selectedHardware) {
    const cardCounts = {};

    if (hardwareMemoryGB.hasOwnProperty(selectedHardware)) {
        const cardMemory = hardwareMemoryGB[selectedHardware];
        const numCards = Math.ceil(estimatedMemoryGB / cardMemory);
        if (numCards > 0) {
            cardCounts[selectedHardware] = numCards;
        }
    } else {
        console.warn(`Selected hardware type "${selectedHardware}" not found in hardwareMemoryGB.`);
        return {};
    }
    return cardCounts;
}


function adjustComputeLoad(currentLoad, factor) {
    const loadLevels = ["低", "中等", "较高", "高", "非常高"];
    let currentIndex = loadLevels.indexOf(currentLoad);
    if (currentIndex === -1) currentIndex = 1;

    let newIndex = Math.round(currentIndex * factor);
    newIndex = Math.max(0, Math.min(loadLevels.length - 1, newIndex));
    return loadLevels[newIndex];
}


function getModelDisplayName(modelType) {
    const modelDisplayNames = {
        'r1_671b': 'DeepSeek R1/V3 671B',
        'r1_1.5b': 'DeepSeek R1 1.5B (蒸馏)',
        'r1_7b': 'DeepSeek R1 7B (蒸馏)',
        'r1_8b': 'DeepSeek R1 8B (蒸馏)',
        'r1_14b': 'DeepSeek R1 14B (蒸馏)',
        'r1_32b': 'DeepSeek R1 32B (蒸馏)',
        'r1_70b': 'DeepSeek R1 70B (蒸馏)'
    };
    return modelDisplayNames[modelType] || modelType;
}


function getDeploymentDisplayName(deploymentMethod) {
    // 部署方式显示名称 (虽然不再使用选择，但函数保留，可能在部署建议中使用)
    const deploymentDisplayNames = {
        'single_card': '单卡部署',
        'multi_card': '多卡部署',
        'multi_machine_multi_card': '多机多卡部署',
        'tensor_parallel': '张量并行',
        'pipeline_parallel': '流水线并行',
        'model_parallel': '模型并行 (通用)'
    };
    return deploymentDisplayNames[deploymentMethod] || deploymentMethod;
}


function getHardwareDisplayName(hardware) {
    const hardwareDisplayNames = {
        'nvidia_h20': 'NVIDIA H20',
        'nvidia_h800': 'NVIDIA H800',
        'nvidia_a800': 'NVIDIA A800',
        'nvidia_l40s': 'NVIDIA L40S',
        'nvidia_a10': 'NVIDIA A10',
        'nvidia_rtx4090': 'NVIDIA RTX 4090',
        'nvidia_a100_40g': 'NVIDIA A100-40G',
        'ascend910b': '华为昇腾910b'
    };
    return hardwareDisplayNames[hardware] || hardware;
}


function getFrameworkDisplayName(framework) {
    const frameworkDisplayNames = {
        'auto': '自动/通用',
        'vllm': 'vLLM',
        'llama_cpp': 'llama.cpp',
        'mindspore': 'MindSpore'
    };
    return frameworkDisplayNames[framework] || framework;
}


function getFineTuningMethodDisplayName(fineTuningMethod) {
    const fineTuningMethodDisplayNames = {
        'inference': '推理',
        'lora': 'LoRA 微调'
    };
    return fineTuningMethodDisplayNames[fineTuningMethod] || fineTuningMethod;
}

function getPrecisionDisplayName(precision) {
    const precisionDisplayNames = {
        'fp8': 'FP8',
        'fp16': 'FP16',
        'bf16': 'BF16',
        'int8': 'INT8',
        'int4': 'INT4'
    };
    return precisionDisplayNames[precision] || precision;
}

// **通用模型参数量定义 (Billion) - 用于非 INT4 精度下的模型大小估算**
const modelParamsBillion_Generic = {
    'r1_671b': 671,
    'r1_1.5b': 1.5,
    'r1_7b': 7,
    'r1_8b': 8,
    'r1_14b': 14,
    'r1_32b': 32,
    'r1_70b': 70,
};