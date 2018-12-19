#include <cusparse_v2.h>

// Global cusparse handle
cusparseHandle_t cusparseHandle;

void CreateCusparseHandle() {
	// create cusparse handle
	cusparseStatus_t status = cusparseCreate(&cusparseHandle);
}
void DestroyCusparseHandle() {
	cusparseDestroy(cusparseHandle);
}