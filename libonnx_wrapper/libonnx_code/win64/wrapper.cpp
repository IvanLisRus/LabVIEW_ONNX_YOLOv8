// wrapper.cpp
// ONNX Runtime Wrapper for LabVIEW (Linux/Windows) под ONNX Runtime 1.24+
//
// Компиляция Linux:
// g++ -shared -fPIC -o libonnx_wrapper.so wrapper.cpp -I./include -L./lib -lonnxruntime -Wl,-rpath,'$ORIGIN' -std=c++17
//
// Компиляция Windows (MinGW-w64 из Linux):
// x86_64-w64-mingw32-g++ -shared -o onnx_wrapper.dll wrapper.cpp -I./include -L./lib -l:onnxruntime.dll -static-libgcc -static-libstdc++ -std=c++17

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <map>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdint>

#ifdef _WIN32
    #define EXPORT_FUNCTION __declspec(dllexport)
    #define PLATFORM_STRDUP _strdup
    #define PLATFORM_SNPRINTF _snprintf
    #include <windows.h>
    #include <codecvt>
    #include <locale>
    
    // Конвертация UTF-8 -> UTF-16 для Windows
    static std::wstring Utf8ToWide(const char* utf8) {
        if (!utf8) return std::wstring();
        int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
        if (len <= 0) return std::wstring();
        std::wstring wide(len, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &wide[0], len);
        wide.resize(len - 1); // убрать завершающий нуль, который добавил MultiByteToWideChar
        return wide;
    }
#else
    #define EXPORT_FUNCTION __attribute__((visibility("default")))
    #define PLATFORM_STRDUP strdup
    #define PLATFORM_SNPRINTF snprintf
#endif

// ============================================================================
// Глобальные переменные
// ============================================================================
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LabVIEW_ONNX_Multi");
Ort::AllocatorWithDefaultOptions allocator;
std::mutex sessionMutex;

// ============================================================================
// Структуры данных
// ============================================================================
struct TensorTypeInfo {
    std::vector<char*> names;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<ONNXTensorElementDataType> elementTypes;
};

struct SessionData {
    Ort::Session* session = nullptr;
    char* modelPath = nullptr;
    TensorTypeInfo inputs;
    TensorTypeInfo outputs;
    bool isLoaded = false;
    char* lastErrorMessage = nullptr;
    std::vector<char*> metadataKeys;
    std::vector<char*> metadataValues;
};

std::map<int, SessionData> sessions;
int nextSessionId = 1;

// ============================================================================
// Вспомогательные функции
// ============================================================================
void SetSessionError(SessionData& data, const char* error) {
    if (data.lastErrorMessage) free(data.lastErrorMessage);
    data.lastErrorMessage = PLATFORM_STRDUP(error);
    fprintf(stderr, "ONNX Error (Session): %s\n", error);
}

void ClearTensorTypeInfo(TensorTypeInfo& info) {
    for (auto name : info.names) free(name);
    info.names.clear();
    info.shapes.clear();
    info.elementTypes.clear();
}

void DestroySessionData(SessionData& data) {
    ClearTensorTypeInfo(data.inputs);
    ClearTensorTypeInfo(data.outputs);
    if (data.session) {
        delete data.session;
        data.session = nullptr;
    }
    if (data.modelPath) {
        free(data.modelPath);
        data.modelPath = nullptr;
    }
    if (data.lastErrorMessage) {
        free(data.lastErrorMessage);
        data.lastErrorMessage = nullptr;
    }
    for (auto key : data.metadataKeys) free(key);
    for (auto val : data.metadataValues) free(val);
    data.metadataKeys.clear();
    data.metadataValues.clear();
    data.isLoaded = false;
}

int InternalLoadModel(SessionData& data, const char* path) {
    try {
        if (data.session) delete data.session;
        ClearTensorTypeInfo(data.inputs);
        ClearTensorTypeInfo(data.outputs);
        for (auto key : data.metadataKeys) free(key);
        for (auto val : data.metadataValues) free(val);
        data.metadataKeys.clear();
        data.metadataValues.clear();

        data.modelPath = PLATFORM_STRDUP(path);

        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);

#ifdef _WIN32
        // На Windows нужно конвертировать путь в wchar_t*
        std::wstring widePath = Utf8ToWide(path);
        data.session = new Ort::Session(env, widePath.c_str(), sessionOptions);
#else
        data.session = new Ort::Session(env, path, sessionOptions);
#endif

        // === ВХОДЫ ===
        size_t inputCount = data.session->GetInputCount();
        for (size_t i = 0; i < inputCount; i++) {
            auto namePtr = data.session->GetInputNameAllocated(i, allocator);
            data.inputs.names.push_back(PLATFORM_STRDUP(namePtr.get()));

            auto typeInfo = data.session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            data.inputs.shapes.push_back(tensorInfo.GetShape());
            data.inputs.elementTypes.push_back(tensorInfo.GetElementType());
        }

        // === ВЫХОДЫ ===
        size_t outputCount = data.session->GetOutputCount();
        for (size_t i = 0; i < outputCount; i++) {
            auto namePtr = data.session->GetOutputNameAllocated(i, allocator);
            data.outputs.names.push_back(PLATFORM_STRDUP(namePtr.get()));

            auto typeInfo = data.session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            data.outputs.shapes.push_back(tensorInfo.GetShape());
            data.outputs.elementTypes.push_back(tensorInfo.GetElementType());
        }

        // === МЕТАДАННЫЕ ===
        try {
            Ort::ModelMetadata modelMetadata = data.session->GetModelMetadata();
            auto keysVec = modelMetadata.GetCustomMetadataMapKeysAllocated(allocator);
            size_t keysCount = keysVec.size();
            
            for (size_t i = 0; i < keysCount; i++) {
                const char* key = keysVec[i].get();
                data.metadataKeys.push_back(PLATFORM_STRDUP(key));
                 
                Ort::AllocatedStringPtr valPtr = modelMetadata.LookupCustomMetadataMapAllocated(key, allocator);
                if (valPtr.get()) {
                    data.metadataValues.push_back(PLATFORM_STRDUP(valPtr.get()));
                } else {
                    data.metadataValues.push_back(PLATFORM_STRDUP(" "));
                }
            }
        } catch (const Ort::Exception & e) {
            fprintf(stderr, "Warning: Could not read model metadata: %s\n", e.what());
        }

        data.isLoaded = true;
        return 0;
    } catch (const Ort::Exception & e) {
        SetSessionError(data, e.what());
        return -1;
    }
}

// ============================================================================
// Экспортируемые функции (C API)
// ============================================================================
extern "C" {

EXPORT_FUNCTION int CreateSession(const char* modelPath) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    int sessionId = nextSessionId++;
    SessionData data;
    sessions[sessionId] = data;
    
    int result = InternalLoadModel(sessions[sessionId], modelPath);
    if (result != 0) {
        DestroySessionData(sessions[sessionId]);
        sessions.erase(sessionId);
        return -1;
    }
    return sessionId;
}

EXPORT_FUNCTION int IsSessionLoaded(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    return sessions[sessionId].isLoaded ? 1 : 0;
}

EXPORT_FUNCTION int UnloadSession(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (data.session) { delete data.session; data.session = nullptr; }
    ClearTensorTypeInfo(data.inputs);
    ClearTensorTypeInfo(data.outputs);
    data.isLoaded = false;
    return 0;
}

EXPORT_FUNCTION int ReloadSession(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.modelPath) return -1;
    return InternalLoadModel(data, data.modelPath);
}

EXPORT_FUNCTION int DestroySession(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    DestroySessionData(sessions[sessionId]);
    sessions.erase(sessionId);
    return 0;
}

EXPORT_FUNCTION void DestroyAllSessions() {
    std::lock_guard<std::mutex> lock(sessionMutex);
    for (auto & pair : sessions) {
        DestroySessionData(pair.second);
    }
    sessions.clear();
    nextSessionId = 1;
}

EXPORT_FUNCTION int GetInputCount(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    if (!sessions[sessionId].session) return 0;
    return (int)sessions[sessionId].session->GetInputCount();
}

EXPORT_FUNCTION int GetOutputCount(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    if (!sessions[sessionId].session) return 0;
    return (int)sessions[sessionId].session->GetOutputCount();
}

EXPORT_FUNCTION int GetInputName(int sessionId, int index, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || index < 0 || index >= (int)data.inputs.names.size()) return -1;
    strncpy(buffer, data.inputs.names[index], bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION int GetOutputName(int sessionId, int index, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || index < 0 || index >= (int)data.outputs.names.size()) return -1;
    strncpy(buffer, data.outputs.names[index], bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION int GetInputShapeDimCount(int sessionId, int inputIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || inputIndex < 0 || inputIndex >= (int)data.inputs.shapes.size()) return 0;
    return (int)data.inputs.shapes[inputIndex].size();
}

EXPORT_FUNCTION int64_t GetInputShapeDim(int sessionId, int inputIndex, int dimIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || inputIndex < 0 || inputIndex >= (int)data.inputs.shapes.size()) return -1;
    if (dimIndex < 0 || dimIndex >= (int)data.inputs.shapes[inputIndex].size()) return -1;
    return data.inputs.shapes[inputIndex][dimIndex];
}

EXPORT_FUNCTION int GetOutputShapeDimCount(int sessionId, int outputIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || outputIndex < 0 || outputIndex >= (int)data.outputs.shapes.size()) return 0;
    return (int)data.outputs.shapes[outputIndex].size();
}

EXPORT_FUNCTION int64_t GetOutputShapeDim(int sessionId, int outputIndex, int dimIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || outputIndex < 0 || outputIndex >= (int)data.outputs.shapes.size()) return -1;
    if (dimIndex < 0 || dimIndex >= (int)data.outputs.shapes[outputIndex].size()) return -1;
    return data.outputs.shapes[outputIndex][dimIndex];
}

EXPORT_FUNCTION int GetInputType(int sessionId, int inputIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || inputIndex < 0 || inputIndex >= (int)data.inputs.elementTypes.size()) return -1;
    return (int)data.inputs.elementTypes[inputIndex];
}

EXPORT_FUNCTION int GetOutputType(int sessionId, int outputIndex) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || outputIndex < 0 || outputIndex >= (int)data.outputs.elementTypes.size()) return -1;
    return (int)data.outputs.elementTypes[outputIndex];
}

EXPORT_FUNCTION int GetInputTypeString(int sessionId, int inputIndex, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || inputIndex < 0 || inputIndex >= (int)data.inputs.elementTypes.size()) return -1;
    
    const char* typeName = "UNKNOWN";
    switch (data.inputs.elementTypes[inputIndex]) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: typeName = "FLOAT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: typeName = "FLOAT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: typeName = "INT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: typeName = "INT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: typeName = "INT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: typeName = "INT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: typeName = "UINT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: typeName = "UINT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: typeName = "UINT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: typeName = "UINT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: typeName = "FLOAT16"; break;
        default: typeName = "UNKNOWN"; break;
    }
    strncpy(buffer, typeName, bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION int GetOutputTypeString(int sessionId, int outputIndex, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || outputIndex < 0 || outputIndex >= (int)data.outputs.elementTypes.size()) return -1;
    
    const char* typeName = "UNKNOWN";
    switch (data.outputs.elementTypes[outputIndex]) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: typeName = "FLOAT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: typeName = "FLOAT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: typeName = "INT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: typeName = "INT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: typeName = "INT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: typeName = "INT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: typeName = "UINT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: typeName = "UINT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: typeName = "UINT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: typeName = "UINT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: typeName = "FLOAT16"; break;
        default: typeName = "UNKNOWN"; break;
    }
    strncpy(buffer, typeName, bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION int RunInference(int sessionId, int inputIndex, float* inputBuffer, int inputSize, 
                 int outputIndex, float* outputBuffer, int outputSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.session) return -1;
    if (inputIndex >= (int)data.inputs.names.size() || outputIndex >= (int)data.outputs.names.size()) return -1;

    try {
        std::vector<int64_t> shape = data.inputs.shapes[inputIndex];
        size_t known_dims_product = 1;
        int unknown_dim_index = -1;
        int unknown_count = 0;
        
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == -1) {
                unknown_dim_index = (int)i;
                unknown_count++;
            } else if (shape[i] > 0) {
                known_dims_product *= shape[i];
            }
        }
        
        if (unknown_count == 1 && unknown_dim_index >= 0 && known_dims_product > 0) {
            if (inputSize % known_dims_product != 0) {
                SetSessionError(data, "Input size does not match expected shape dimensions");
                return -1;
            }
            shape[unknown_dim_index] = inputSize / known_dims_product;
        } else if (unknown_count > 1) {
            SetSessionError(data, "Multiple dynamic dimensions (-1) in shape are not supported");
            return -1;
        } else if (unknown_count == 0) {
            size_t expected_size = known_dims_product;
            if ((size_t)inputSize != expected_size) {
                char error_msg[256];
                PLATFORM_SNPRINTF(error_msg, sizeof(error_msg), 
                          "Input size mismatch: expected %zu, got %d", 
                          expected_size, inputSize);
                SetSessionError(data, error_msg);
                return -1;
            }
        }

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputBuffer, inputSize, shape.data(), shape.size());

        const char* inputNamesArr[] = {data.inputs.names[inputIndex]};
        const char* outputNamesArr[] = {data.outputs.names[outputIndex]};

        auto outputTensors = data.session->Run(
            Ort::RunOptions{nullptr}, 
            inputNamesArr, &inputTensor, 1, 
            outputNamesArr, 1);
        
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        memcpy(outputBuffer, outputData, outputSize * sizeof(float));
        
        return 0;
        
    } catch (const Ort::Exception & e) {
        SetSessionError(data, e.what());
        return -1;
    } catch (const std::exception & e) {
        SetSessionError(data, e.what());
        return -1;
    } catch (...) {
        SetSessionError(data, "Unknown error in RunInference");
        return -1;
    }
}

// === МЕТАДАННЫЕ ===
EXPORT_FUNCTION int GetMetadataCount(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return 0;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded) return 0;
    return (int)data.metadataKeys.size();
}

EXPORT_FUNCTION int GetMetadataKey(int sessionId, int index, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded || index < 0 || index >= (int)data.metadataKeys.size()) return -1;
    strncpy(buffer, data.metadataKeys[index], bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION int GetMetadataValue(int sessionId, const char* key, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    if (!data.isLoaded) return -1;
    
    for (size_t i = 0; i < data.metadataKeys.size(); i++) {
        if (strcmp(data.metadataKeys[i], key) == 0) {
            strncpy(buffer, data.metadataValues[i], bufferSize - 1);
            buffer[bufferSize - 1] = '\0';
            return 0;
        }
    }
    return -1;
}

// === ОШИБКИ ===
EXPORT_FUNCTION int GetLastErrorMessage(int sessionId, char* buffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return -1;
    SessionData & data = sessions[sessionId];
    
    if (!data.lastErrorMessage) {
        strncpy(buffer, "No error", bufferSize - 1);
        buffer[bufferSize - 1] = '\0';
        return 0;
    }
    
    strncpy(buffer, data.lastErrorMessage, bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return 0;
}

EXPORT_FUNCTION void ClearLastError(int sessionId) {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (sessions.find(sessionId) == sessions.end()) return;
    SessionData & data = sessions[sessionId];
    if (data.lastErrorMessage) {
        free(data.lastErrorMessage);
        data.lastErrorMessage = nullptr;
    }
}

} // extern "C"
