# YALS 思考解析與工具解析改進方案

## 1. 當前問題分析

### 1.1 模板元數據支持不足

**當前實現** (`common/templating.ts`):
```typescript
const TemplateMetadataSchema = z.object({
    stop_strings: z.array(z.string()).default([]),
    tool_start: z.string().optional(),
});
```

**問題**:
- 只支持 `stop_strings` 和 `tool_start`
- 缺少思考標籤的開始/結束標記
- 缺少工具調用的結束標記
- 無法支持不同模型的自定義格式

### 1.2 工具解析格式單一

**當前實現** (`api/OAI/utils/tools.ts`):
- 只支持 `<tool_call>...</tool_call>` XML格式
- 支持JSON格式
- 缺少對以下格式的支持:
  - Hermes格式
  - Llama 3.x原生格式
  - Qwen格式
  - Mistral格式
  - DeepSeek格式

### 1.3 思考解析完全缺失

當前代碼中沒有任何思考內容的解析支持，這意味著:
- 無法處理 `<think>...</think>` 標籤
- 無法處理 `<thinking>...</thinking>` 標籤
- 無法正確計算思考token和輸出token
- 無法在API響應中返回思考內容

### 1.4 Token計算問題

- 流式輸出時思考token和工具token的計算不準確
- 無法區分思考token和輸出token
- 使用統計中缺少thinking_tokens字段

## 2. 改進方案設計

### 2.1 擴展模板元數據

**新的 `TemplateMetadata` 結構**:

```typescript
const TemplateMetadataSchema = z.object({
    // 停止字符串
    stop_strings: z.array(z.string()).default([]),

    // 工具調用相關
    tool_start: z.string().optional(),        // 工具調用開始標記
    tool_end: z.string().optional(),          // 工具調用結束標記
    tool_format: z.enum([                     // 工具格式類型
        "xml",           // <tool_call>...</tool_call>
        "json",          // {"name": ..., "arguments": ...}
        "hermes",        // <tool_call>{"name": ...}</tool_call>
        "llama3",        // <|python_tag|>...
        "qwen",          // <tool_call>...</tool_call> (Qwen風格)
        "mistral",       // [TOOL_CALLS]...
        "deepseek",      // DeepSeek格式
        "generic"        // 通用格式
    ]).default("xml"),

    // 思考/推理相關
    thinking_start: z.string().optional(),    // 思考開始標記
    thinking_end: z.string().optional(),      // 思考結束標記
    thinking_format: z.enum([                 // 思考格式類型
        "deepseek_r1",   // <think>...</think>
        "qwen3",         // <think>...</think> (默認開啟)
        "granite",       // 需要顯式開啟
        "generic"        // 通用 <thinking>...</thinking>
    ]).optional(),

    // 功能標誌
    supports_parallel_tools: z.boolean().default(false),
    supports_interleaved_thinking: z.boolean().default(false),
});
```

### 2.2 創建統一的解析器架構

**參考vLLM的設計模式**:

```typescript
// 基礎解析器接口
interface ContentParser {
    // 流式解析
    processStreaming(text: string): ParsedContent;
    flush(): ParsedContent;

    // 完整文本解析
    parseComplete(text: string): ParsedContent;
}

interface ParsedContent {
    text: string;              // 普通文本內容
    thinking?: string;         // 思考內容
    toolCalls?: ToolCall[];    // 工具調用
    isComplete: boolean;       // 是否解析完成
}

// 思考解析器
interface ReasoningParser extends ContentParser {
    extractReasoningContent(text: string): {
        reasoning: string;
        content: string;
    };
}

// 工具解析器
interface ToolCallParser extends ContentParser {
    extractToolCalls(text: string): {
        toolCalls: ToolCall[];
        content: string;
    };
}
```

### 2.3 實現具體解析器

#### 2.3.1 DeepSeek R1 思考解析器

```typescript
class DeepSeekR1ReasoningParser implements ReasoningParser {
    private thinkingStart = "<think>";
    private thinkingEnd = "</think>";

    extractReasoningContent(text: string) {
        const regex = /<think>([\s\S]*?)<\/think>/g;
        let reasoning = "";
        let content = text;

        let match;
        while ((match = regex.exec(text)) !== null) {
            reasoning += match[1];
            content = content.replace(match[0], "");
        }

        return { reasoning: reasoning.trim(), content: content.trim() };
    }

    // 流式處理
    processStreaming(text: string): ParsedContent {
        // 實現流式解析邏輯
    }
}
```

#### 2.3.2 Hermes 工具解析器

```typescript
class HermesToolCallParser implements ToolCallParser {
    private toolCallStart = "<tool_call>";
    private toolCallEnd = "</tool_call>";

    extractToolCalls(text: string) {
        const regex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
        const toolCalls: ToolCall[] = [];
        let content = text;

        let match;
        while ((match = regex.exec(text)) !== null) {
            try {
                const parsed = JSON.parse(match[1]);
                toolCalls.push(this.normalizeToolCall(parsed));
                content = content.replace(match[0], "");
            } catch {
                // 處理解析錯誤
            }
        }

        return { toolCalls, content: content.trim() };
    }
}
```

#### 2.3.3 Llama 3.x 原生工具解析器

```typescript
class Llama3ToolCallParser implements ToolCallParser {
    private pythonTag = "<|python_tag|>";

    extractToolCalls(text: string) {
        // Llama 3.x 使用特殊格式
        // {"name": "function_name", "parameters": {...}}
        // 處理內置工具: wolfram_alpha, web_search, code_interpreter
    }
}
```

### 2.4 統一的內容處理器

```typescript
class UnifiedContentProcessor {
    private reasoningParser?: ReasoningParser;
    private toolCallParser?: ToolCallParser;
    private metadata: TemplateMetadata;

    constructor(metadata: TemplateMetadata) {
        this.metadata = metadata;
        this.initializeParsers();
    }

    private initializeParsers() {
        // 根據metadata初始化解析器
        if (this.metadata.thinking_format) {
            this.reasoningParser = ReasoningParserFactory.create(
                this.metadata.thinking_format
            );
        }

        if (this.metadata.tool_format) {
            this.toolCallParser = ToolCallParserFactory.create(
                this.metadata.tool_format
            );
        }
    }

    process(text: string): ProcessedContent {
        let result: ProcessedContent = {
            text,
            thinking: undefined,
            toolCalls: [],
        };

        // 先提取思考內容
        if (this.reasoningParser) {
            const { reasoning, content } = this.reasoningParser
                .extractReasoningContent(text);
            result.thinking = reasoning;
            result.text = content;
        }

        // 再提取工具調用
        if (this.toolCallParser) {
            const { toolCalls, content } = this.toolCallParser
                .extractToolCalls(result.text);
            result.toolCalls = toolCalls;
            result.text = content;
        }

        return result;
    }
}
```

### 2.5 更新API響應格式

#### OpenAI兼容API擴展

```typescript
interface ChatCompletionResponse {
    // ... 現有字段

    // 新增字段
    reasoning_content?: string;  // 思考內容
}

interface UsageStats {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;

    // 新增字段
    reasoning_tokens?: number;   // 思考token數
    cached_tokens?: number;      // 緩存token數
}
```

#### Anthropic兼容API擴展

```typescript
interface AnthropicContentBlock {
    type: "text" | "tool_use" | "thinking";  // 新增thinking類型
    // ...
}
```

### 2.6 配置系統擴展

**更新 `config_sample.yml`**:

```yaml
model:
  # ... 現有配置

  # 新增解析器配置
  reasoning_parser: "deepseek_r1"  # 或 "qwen3", "granite", "generic"
  tool_parser: "hermes"            # 或 "llama3", "mistral", "generic"

  # 思考模式配置
  thinking:
    enabled: true                  # 是否啟用思考解析
    forced_open: false             # 強制始終輸出思考內容
    include_in_response: true      # 是否在響應中包含思考內容

  # 工具調用配置
  tool_calling:
    parallel_enabled: false        # 是否啟用並行工具調用
    auto_detect: true              # 自動檢測工具調用格式
```

## 3. 實現路線圖

### 階段1: 基礎架構 (核心改進)

1. **擴展 `TemplateMetadata`** (`common/templating.ts`)
   - 添加思考和工具相關的元數據字段
   - 更新AST解析邏輯以提取新字段

2. **創建解析器基礎架構** (`common/parsers/`)
   - `base.ts` - 基礎接口定義
   - `factory.ts` - 解析器工廠
   - `registry.ts` - 解析器註冊管理

3. **更新類型定義** (`types/`)
   - 添加新的響應類型
   - 更新使用統計類型

### 階段2: 實現具體解析器

1. **思考解析器** (`common/parsers/reasoning/`)
   - `deepseek_r1.ts` - DeepSeek R1格式
   - `qwen3.ts` - Qwen3格式
   - `generic.ts` - 通用格式

2. **工具解析器** (`common/parsers/tools/`)
   - `hermes.ts` - Hermes格式
   - `llama3.ts` - Llama 3.x格式
   - `mistral.ts` - Mistral格式
   - `generic.ts` - 通用XML/JSON格式

### 階段3: 集成到API層

1. **更新 OAI API** (`api/OAI/`)
   - 修改 `chatCompletion.ts` 使用新解析器
   - 更新響應生成邏輯

2. **更新 Anthropic API** (`api/anthropic/`)
   - 修改 `messages.ts` 使用新解析器
   - 支持thinking content block

3. **更新流式處理**
   - 正確處理流式輸出中的思考內容
   - 準確計算各類token

### 階段4: 測試和文檔

1. 編寫單元測試
2. 編寫集成測試
3. 更新API文檔
4. 更新配置文檔

## 4. 參考資源

- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [vLLM Interleaved Thinking](https://docs.vllm.ai/en/latest/features/interleaved_thinking/)
- [llama.cpp Function Calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [llama.cpp Chat Templates](https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)

## 5. 兼容性考慮

### 5.1 向後兼容

- 所有新字段都是可選的
- 默認行為與當前實現保持一致
- 現有模板無需修改即可繼續使用

### 5.2 模型兼容性

| 模型系列 | 思考格式 | 工具格式 | 注意事項 |
|---------|---------|---------|---------|
| DeepSeek R1 | deepseek_r1 | deepseek | 需要 `thinking=True` |
| Qwen3 | qwen3 | qwen | 默認開啟思考 |
| Qwen3-Coder | qwen3 | qwen_coder | 自定義XML格式 |
| Llama 3.x | - | llama3 | 支持內置工具 |
| Hermes 2/3 | generic | hermes | - |
| Mistral | - | mistral | - |
| Granite 3.2 | granite | generic | 需要顯式開啟 |

## 6. 預期效果

實現此方案後:

1. **更好的模板支持**: 可以從模板中自動提取思考和工具相關的配置
2. **準確的Token計算**: 正確區分和計算思考token、輸出token、工具token
3. **多格式支持**: 支持各種主流模型的工具和思考格式
4. **更好的API兼容性**: 與OpenAI和Anthropic API更好地對接
5. **靈活的配置**: 用戶可以根據需要配置解析器行為
