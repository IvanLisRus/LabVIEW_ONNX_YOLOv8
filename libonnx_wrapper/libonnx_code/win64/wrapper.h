// wrapper.h
// ONNX Runtime Multi-Session Wrapper for LabVIEW (Linux)
// API Reference

#ifndef ONNX_WRAPPER_H
#define ONNX_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Создать новую сессию и загрузить модель
 * @param modelPath Путь к файлу .onnx
 * @return SessionID (>=1) или -1 при ошибке
 */
int CreateSession(const char* modelPath);

/**
 * Проверить, загружена ли сессия в память
 * @param sessionId ID сессии
 * @return 1 если загружена, 0 если нет
 */
int IsSessionLoaded(int sessionId);

/**
 * Выгрузить сессию из памяти (освободить веса, сохранить ID и путь)
 * @param sessionId ID сессии
 * @return 0 при успехе, -1 при ошибке
 */
int UnloadSession(int sessionId);

/**
 * Перезагрузить сессию по сохраненному пути
 * @param sessionId ID сессии
 * @return 0 при успехе, -1 при ошибке
 */
int ReloadSession(int sessionId);

/**
 * Полностью удалить сессию (очистить память и ID)
 * @param sessionId ID сессии
 * @return 0 при успехе, -1 при ошибке
 */
int DestroySession(int sessionId);

/**
 * Получить количество входов модели
 * @param sessionId ID сессии
 * @return Количество входов
 */
int GetInputCount(int sessionId);

/**
 * Получить количество выходов модели
 * @param sessionId ID сессии
 * @return Количество выходов
 */
int GetOutputCount(int sessionId);

/**
 * Получить имя входа по индексу
 * @param sessionId ID сессии
 * @param index Индекс входа (0-based)
 * @param buffer Буфер для строки
 * @param bufferSize Размер буфера
 * @return 0 при успехе, -1 при ошибке
 */
int GetInputName(int sessionId, int index, char* buffer, int bufferSize);

/**
 * Получить имя выхода по индексу
 * @param sessionId ID сессии
 * @param index Индекс выхода (0-based)
 * @param buffer Буфер для строки
 * @param bufferSize Размер буфера
 * @return 0 при успехе, -1 при ошибке
 */
int GetOutputName(int sessionId, int index, char* buffer, int bufferSize);

/**
 * Получить количество размерностей входа
 * @param sessionId ID сессии
 * @param inputIndex Индекс входа
 * @return Количество размерностей (например, 2 для [1, 10])
 */
int GetInputShapeDimCount(int sessionId, int inputIndex);

/**
 * Получить значение размерности входа
 * @param sessionId ID сессии
 * @param inputIndex Индекс входа
 * @param dimIndex Индекс размерности (0-based)
 * @return Значение размерности (например, 10)
 */
int64_t GetInputShapeDim(int sessionId, int inputIndex, int dimIndex);

/**
 * Получить количество размерностей выхода
 * @param sessionId ID сессии
 * @param outputIndex Индекс выхода
 * @return Количество размерностей
 */
int GetOutputShapeDimCount(int sessionId, int outputIndex);

/**
 * Получить значение размерности выхода
 * @param sessionId ID сессии
 * @param outputIndex Индекс выхода
 * @param dimIndex Индекс размерности (0-based)
 * @return Значение размерности
 */
int64_t GetOutputShapeDim(int sessionId, int outputIndex, int dimIndex);

/**
 * Выполнить инференс
 * @param sessionId ID сессии
 * @param inputIndex Индекс входа (обычно 0)
 * @param inputBuffer Массив входных данных (float32)
 * @param inputSize Количество элементов во входном массиве
 * @param outputIndex Индекс выхода (обычно 0)
 * @param outputBuffer Массив для выходных данных (float32)
 * @param outputSize Размер выходного буфера
 * @return 0 при успехе, -1 при ошибке
 */
int RunInference(int sessionId, int inputIndex, float* inputBuffer, int inputSize, 
                 int outputIndex, float* outputBuffer, int outputSize);

#ifdef __cplusplus
}
#endif

#endif // ONNX_WRAPPER_H
