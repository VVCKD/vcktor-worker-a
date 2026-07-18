# VCKTOR Worker A (RunPod Serverless)

커스텀 업로드 pth/index 모델 전용 RVC 음성변환 워커.
기본 8모델은 기존 워커가 처리하고, 이 워커는 `api.vvckd.ai/static/models/<name>/<name>.pth(.index)` 를 받아 변환한다.

## Job I/O (기존 백엔드 규약과 동일)
- input: `{ model_id, params, input_files: [presigned URL...] }`
- output: `{ files: [{ status, output_url(=vcktor/outputs/...), filename }] }`

## RunPod 엔드포인트 환경변수
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (vcktor/outputs/* PutObject 권한 IAM 키)
- `S3_BUCKET_NAME` = vvckd-vcktor-prod-458207137452
- `AWS_REGION` = ap-northeast-2
- (선택) `MODEL_STATIC_BASE` 기본 https://api.vvckd.ai/static/models
