# 仓库知识点自动映射（草稿）

根目录：/home/cx857322378/shengteng

## 配置与入口

- backend/api/auth_routes.py
- backend/api/websocket.py
- backend/core/response_models.py
- backend/main.py
- backend/quick_start.py
- backend/test_server.py
- backend/utils/config.py
- backend/utils/security.py
- docs/knowledge-map-auto.md
- frontend/package-lock.json
- health_check.py
- quick_start_learning.py
- requirements.txt
- tools/scan_repo.py
- training/debug_dataset.py
- training/debug_decoder.py
- training/evaluator.py
- training/requirements.txt
- training/start_training.py
- training/test_basic.py
- training/test_model.py
- training/train_tfnet.py
- training/video_test.py
- training/vocab_diagnosis.py

## 数据与预处理

- README.md
- backend/api/system_routes.py
- backend/core/config_manager.py
- backend/services/diffusion_slp_service.py
- backend/services/mediapipe_service.py
- backend/services/privacy_service.py
- backend/services/sign_recognition_service.py
- backend/utils/config.py
- backend/utils/file_manager.py
- docs/development-guide.md
- docs/development.md
- docs/knowledge-map-auto.md
- frontend/package-lock.json
- health_check.py
- requirements-tfnet.txt
- requirements.txt
- training/QUICK_START_GUIDE.md
- training/data_processor.py
- training/modules.py
- training/requirements.txt
- training/start_training.py
- training/test_basic.py
- training/test_model.py
- training/train_tfnet.py
- training/utils.py
- training/video_test.py

## 模型定义

- backend/services/cslr_service.py
- backend/services/diffusion_slp_service.py
- docs/development-guide.md
- docs/development.md
- docs/knowledge-map-auto.md
- training/decoder.py
- training/docs/训练流程说明.md
- training/modules.py
- training/tfnet_model.py
- training/train_tfnet.py

## 训练循环

- backend/services/cslr_service.py
- backend/services/diffusion_slp_service.py
- backend/services/federated_learning_service.py
- backend/services/multimodal_sensor_service.py
- backend/services/privacy_service.py
- backend/services/sign_recognition_service.py
- docs/development-guide.md
- docs/tfnet-integration-guide.md
- training/config_manager.py
- training/configs/base_default.json
- training/configs/cecsl_training_config.json
- training/configs/enhanced_default.json
- training/configs/optimal_default.json
- training/configs/tfnet_config.json
- training/debug_decoder.py
- training/evaluator.py
- training/test_basic.py
- training/test_model.py
- training/train_tfnet.py
- training/video_test.py

## 损失与评估

- backend/utils/config.py
- docs/knowledge-map-auto.md
- training/test_model.py
- training/video_test.py

## 学习率与训练策略

- training/configs/base_default.json
- training/configs/cecsl_training_config.json
- training/configs/enhanced_default.json
- training/configs/optimal_default.json

## 昇腾与分布式

- backend/services/diffusion_slp_service.py
- backend/services/federated_learning_service.py
- backend/services/multimodal_sensor_service.py
- backend/services/privacy_service.py
- docs/development-guide.md

## 推理与服务

- README.md
- backend/Dockerfile
- backend/api/auth_routes.py
- backend/api/learning_routes.py
- backend/api/system_routes.py
- backend/api/websocket.py
- backend/api/websocket_routes.py
- backend/core/websocket_manager.py
- backend/main.py
- backend/middleware/rate_limiting.py
- backend/middleware/security_headers.py
- backend/services/cslr_service.py
- backend/services/diffusion_slp_service.py
- backend/services/multimodal_sensor_service.py
- backend/services/privacy_service.py
- backend/services/realtime_recognition_service.py
- backend/services/sign_recognition_service.py
- backend/test_server.py
- backend/utils/config.py
- backend/utils/file_manager.py
- backend/utils/monitoring.py
- backend/utils/security.py
- backend/集成报告.md
- docs/deployment.md
- docs/development-guide.md
- docs/development.md
- docs/knowledge-map-auto.md
- health_check.py
- quick_start_learning.py
- requirements-tfnet.txt
- requirements.txt
- training/evaluator.py

## 联邦学习

- backend/services/federated_learning_service.py
- backend/utils/config.py

## 部署与运维

- README.md
- backend/Dockerfile
- docs/deployment.md
- docs/development-guide.md
- docs/development.md
- docs/knowledge-map-auto.md
- docs/tfnet-integration-guide.md
- frontend/Dockerfile
- health_check.py
- quick_start_learning.py
- tools/scan_repo.py

## 文档与指南

- .github/chatmodes/test.chatmode.md
- README.md
- USER_MANUAL.md
- backend/Dockerfile
- backend/__init__.py
- backend/api/__init__.py
- backend/api/auth_routes.py
- backend/api/learning_routes.py
- backend/api/system_routes.py
- backend/api/websocket.py
- backend/api/websocket_routes.py
- backend/core/cache_manager.py
- backend/core/config_manager.py
- backend/core/database_manager.py
- backend/core/migration_manager.py
- backend/core/response_models.py
- backend/core/service_manager.py
- backend/core/websocket_manager.py
- backend/main.py
- backend/middleware/rate_limiting.py
- backend/middleware/security_headers.py
- backend/quick_start.py
- backend/services/__init__.py
- backend/services/achievement_service.py
- backend/services/course_management_service.py
- backend/services/cslr_service.py
- backend/services/diffusion_slp_service.py
- backend/services/enhanced_learning_service.py
- backend/services/federated_learning_service.py
- backend/services/haptic_service.py
- backend/services/learning_training_service.py
- backend/services/mediapipe_service.py
- backend/services/multimodal_sensor_service.py
- backend/services/privacy_service.py
- backend/services/realtime_recognition_service.py
- backend/services/sign_recognition_service.py
- backend/test_server.py
- backend/utils/__init__.py
- backend/utils/cache.py
- backend/utils/config.py
- backend/utils/database.py
- backend/utils/file_manager.py
- backend/utils/logger.py
- backend/utils/monitoring.py
- backend/utils/security.py
- backend/集成报告.md
- docs/3d-error-handling-solution.md
- docs/avatar-error-fixes.md
- docs/avatar-human-upgrade.md
- docs/avatar-upgrade-solution.md
- docs/deployment.md
- docs/development-guide.md
- docs/development.md
- docs/knowledge-map-auto.md
- docs/tfnet-integration-guide.md
- docs/user-guide.md
- frontend/Dockerfile
- frontend/src/components/DetailedHandModel_Optimization_Summary.md
- health_check.py
- quick_start_learning.py
- requirements-tfnet.txt
- requirements.txt
- tools/scan_repo.py
- training/QUICK_START_GUIDE.md
- training/README.md
- training/config_manager.py
- training/data_processor.py
- training/debug_dataset.py
- training/debug_decoder.py
- training/decoder.py
- training/docs/训练流程说明.md
- training/evaluator.py
- training/modules.py
- training/requirements.txt
- training/start_training.py
- training/test_basic.py
- training/test_model.py
- training/tfnet_model.py
- training/train_tfnet.py
- training/utils.py
- training/video_test.py
- training/vocab_diagnosis.py

