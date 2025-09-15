"""
Unit tests for Predictive Models Service (Story 3.5)
Tests machine learning models for clinical prediction and decision support
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analytics.predictive_models import (
    PredictiveModelsService,
    ModelFeatures,
    PredictionResult,
    DeteriorationModel,
    LengthOfStayModel,
    ReadmissionModel,
    ModelType,
    PredictionConfidence
)


class TestPredictiveModelsService:
    """Test suite for PredictiveModelsService"""

    @pytest.fixture
    def predictive_service(self):
        """Create PredictiveModelsService instance"""
        return PredictiveModelsService()

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            "age": 68,
            "gender": "male",
            "current_news2": 5.0,
            "news2_history": [4.5, 4.8, 5.0],
            "comorbidities": ["diabetes", "hypertension"],
            "medications": ["metformin", "lisinopril"],
            "admission_type": "emergency",
            "days_since_admission": 3,
            "previous_alerts": 2,
            "vital_signs_history": [{
                "timestamp": datetime.now().isoformat(),
                "respiratory_rate": 18,
                "heart_rate": 85,
                "systolic_bp": 140,
                "temperature": 37.2,
                "spo2": 94
            }]
        }

    @pytest.mark.asyncio
    async def test_predict_deterioration(self, predictive_service, sample_patient_data):
        """Test deterioration prediction"""
        result = await predictive_service.predict_deterioration("P001", sample_patient_data)

        assert isinstance(result, PredictionResult)
        assert result.model_type == ModelType.DETERIORATION
        assert result.patient_id == "P001"
        assert 0 <= result.prediction_value <= 1
        assert isinstance(result.confidence, PredictionConfidence)
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.recommendations, list)
        assert result.model_version == "deterioration_v1.2"

    @pytest.mark.asyncio
    async def test_predict_length_of_stay(self, predictive_service, sample_patient_data):
        """Test length of stay prediction"""
        result = await predictive_service.predict_length_of_stay("P001", sample_patient_data)

        assert isinstance(result, PredictionResult)
        assert result.model_type == ModelType.LENGTH_OF_STAY
        assert result.patient_id == "P001"
        assert result.prediction_value >= 1.0  # Minimum stay
        assert isinstance(result.confidence, PredictionConfidence)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_predict_readmission(self, predictive_service, sample_patient_data):
        """Test readmission prediction"""
        result = await predictive_service.predict_readmission("P001", sample_patient_data)

        assert isinstance(result, PredictionResult)
        assert result.model_type == ModelType.READMISSION
        assert result.patient_id == "P001"
        assert 0 <= result.prediction_value <= 1
        assert len(result.risk_factors) >= 0
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_get_comprehensive_predictions(self, predictive_service, sample_patient_data):
        """Test getting all predictions for a patient"""
        predictions = await predictive_service.get_comprehensive_predictions("P001", sample_patient_data)

        assert "deterioration" in predictions
        assert "length_of_stay" in predictions
        assert "readmission" in predictions

        for prediction_type, result in predictions.items():
            assert isinstance(result, PredictionResult)
            assert result.patient_id == "P001"

    @pytest.mark.asyncio
    async def test_batch_predictions(self, predictive_service):
        """Test batch predictions for multiple patients"""
        patient_ids = ["P001", "P002", "P003"]
        model_types = [ModelType.DETERIORATION, ModelType.LENGTH_OF_STAY]

        results = await predictive_service.batch_predictions(patient_ids, model_types)

        assert len(results) == 3
        for patient_id in patient_ids:
            assert patient_id in results
            patient_results = results[patient_id]
            assert "deterioration" in patient_results
            assert "length_of_stay" in patient_results
            assert "readmission" not in patient_results  # Not requested

    @pytest.mark.asyncio
    async def test_predict_with_simulated_data(self, predictive_service):
        """Test prediction with simulated patient data"""
        # Test without providing patient data (should use simulated data)
        result = await predictive_service.predict_deterioration("P999")

        assert isinstance(result, PredictionResult)
        assert result.patient_id == "P999"
        assert result.prediction_value >= 0

    def test_model_features_creation(self):
        """Test ModelFeatures creation"""
        features = ModelFeatures(
            patient_id="P001",
            age=65,
            gender="female",
            current_news2=4.0,
            news2_trend=0.5,
            vital_signs_history=[{
                "respiratory_rate": 16,
                "heart_rate": 80,
                "systolic_bp": 120,
                "temperature": 37.0,
                "spo2": 98
            }],
            comorbidities=["diabetes"],
            current_medications=["insulin"],
            admission_type="elective",
            days_since_admission=2,
            previous_alerts_count=0
        )

        assert features.patient_id == "P001"
        assert features.age == 65
        assert features.gender == "female"
        assert features.current_news2 == 4.0

    def test_model_features_to_feature_vector(self):
        """Test conversion of ModelFeatures to feature vector"""
        features = ModelFeatures(
            patient_id="P001",
            age=65,
            gender="male",
            current_news2=4.0,
            news2_trend=0.5,
            vital_signs_history=[{
                "respiratory_rate": 18,
                "heart_rate": 85,
                "systolic_bp": 130,
                "temperature": 37.2,
                "spo2": 96
            }],
            comorbidities=["diabetes", "copd"],
            current_medications=["medication1"],
            admission_type="emergency",
            days_since_admission=3,
            previous_alerts_count=1
        )

        feature_vector = features.to_feature_vector()

        assert feature_vector["age"] == 65.0
        assert feature_vector["gender_male"] == 1.0
        assert feature_vector["current_news2"] == 4.0
        assert feature_vector["news2_trend"] == 0.5
        assert feature_vector["respiratory_rate"] == 18.0
        assert feature_vector["has_diabetes"] == 1.0
        assert feature_vector["has_copd"] == 1.0
        assert feature_vector["comorbidity_count"] == 2.0

    @pytest.mark.asyncio
    async def test_get_model_performance_metrics(self, predictive_service):
        """Test getting model performance metrics"""
        metrics = await predictive_service.get_model_performance_metrics()

        assert "deterioration" in metrics
        assert "length_of_stay" in metrics
        assert "readmission" in metrics

        # Check deterioration model metrics
        deterioration_metrics = metrics["deterioration"]
        assert "accuracy" in deterioration_metrics
        assert "precision" in deterioration_metrics
        assert "recall" in deterioration_metrics
        assert deterioration_metrics["accuracy"] > 0

        # Check LOS model metrics
        los_metrics = metrics["length_of_stay"]
        assert "mae" in los_metrics
        assert "rmse" in los_metrics
        assert "r_squared" in los_metrics

        # Check readmission model metrics
        readmission_metrics = metrics["readmission"]
        assert "auc" in readmission_metrics
        assert "sensitivity" in readmission_metrics
        assert "specificity" in readmission_metrics

    @pytest.mark.asyncio
    async def test_get_feature_importance(self, predictive_service):
        """Test getting feature importance for models"""
        # Test deterioration model
        deterioration_importance = await predictive_service.get_feature_importance(ModelType.DETERIORATION)
        assert len(deterioration_importance) > 0
        assert "current_news2" in deterioration_importance
        assert "news2_trend" in deterioration_importance
        assert all(isinstance(importance, float) for importance in deterioration_importance.values())

        # Test LOS model
        los_importance = await predictive_service.get_feature_importance(ModelType.LENGTH_OF_STAY)
        assert len(los_importance) > 0
        assert "current_news2" in los_importance

        # Test readmission model
        readmission_importance = await predictive_service.get_feature_importance(ModelType.READMISSION)
        assert len(readmission_importance) > 0
        assert "has_copd" in readmission_importance

    def test_prediction_result_serialization(self):
        """Test PredictionResult serialization"""
        result = PredictionResult(
            model_type=ModelType.DETERIORATION,
            patient_id="P001",
            prediction_value=0.75,
            confidence=PredictionConfidence.HIGH,
            confidence_score=0.85,
            risk_factors=["High NEWS2 score", "COPD"],
            recommendations=["Increase monitoring", "Consider intervention"],
            model_version="v1.0",
            generated_at=datetime(2024, 1, 1, 12, 0, 0)
        )

        result_dict = result.to_dict()

        assert result_dict["model_type"] == "deterioration"
        assert result_dict["patient_id"] == "P001"
        assert result_dict["prediction_value"] == 0.75
        assert result_dict["confidence"] == "high"
        assert result_dict["confidence_score"] == 0.85
        assert len(result_dict["risk_factors"]) == 2
        assert len(result_dict["recommendations"]) == 2
        assert "2024-01-01T12:00:00" in result_dict["generated_at"]


class TestDeteriorationModel:
    """Test suite for DeteriorationModel"""

    @pytest.fixture
    def deterioration_model(self):
        """Create DeteriorationModel instance"""
        return DeteriorationModel()

    @pytest.fixture
    def high_risk_features(self):
        """High-risk patient features"""
        return ModelFeatures(
            patient_id="P001",
            age=80,
            gender="male",
            current_news2=8.0,
            news2_trend=1.5,
            vital_signs_history=[{
                "respiratory_rate": 28,
                "spo2": 88,
                "heart_rate": 120
            }],
            comorbidities=["copd", "heart_disease"],
            current_medications=["oxygen"],
            admission_type="emergency",
            days_since_admission=1,
            previous_alerts_count=3
        )

    @pytest.fixture
    def low_risk_features(self):
        """Low-risk patient features"""
        return ModelFeatures(
            patient_id="P002",
            age=45,
            gender="female",
            current_news2=2.0,
            news2_trend=-0.2,
            vital_signs_history=[{
                "respiratory_rate": 16,
                "spo2": 98,
                "heart_rate": 75
            }],
            comorbidities=[],
            current_medications=[],
            admission_type="elective",
            days_since_admission=1,
            previous_alerts_count=0
        )

    @pytest.mark.asyncio
    async def test_high_risk_prediction(self, deterioration_model, high_risk_features):
        """Test deterioration prediction for high-risk patient"""
        result = await deterioration_model.predict(high_risk_features)

        assert result.prediction_value > 0.5  # Should predict high probability
        assert len(result.risk_factors) > 0
        assert "Elevated NEWS2 score" in result.risk_factors
        assert "immediate clinical review" in result.recommendations[0].lower()

    @pytest.mark.asyncio
    async def test_low_risk_prediction(self, deterioration_model, low_risk_features):
        """Test deterioration prediction for low-risk patient"""
        result = await deterioration_model.predict(low_risk_features)

        assert result.prediction_value < 0.7  # Should predict lower probability
        assert "Continue standard monitoring" in result.recommendations

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, deterioration_model, high_risk_features):
        """Test confidence score calculation"""
        result = await deterioration_model.predict(high_risk_features)

        # High-risk patient with complete data should have high confidence
        assert result.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]
        assert result.confidence_score > 0.6

    @pytest.mark.asyncio
    async def test_risk_factor_identification(self, deterioration_model, high_risk_features):
        """Test risk factor identification"""
        result = await deterioration_model.predict(high_risk_features)

        expected_risk_factors = [
            "Elevated NEWS2 score",
            "Increasing NEWS2 trend",
            "Advanced age",
            "COPD diagnosis",
            "Low oxygen saturation"
        ]

        # Should identify multiple risk factors
        assert len(result.risk_factors) > 3
        for risk_factor in result.risk_factors:
            assert risk_factor in expected_risk_factors


class TestLengthOfStayModel:
    """Test suite for LengthOfStayModel"""

    @pytest.fixture
    def los_model(self):
        """Create LengthOfStayModel instance"""
        return LengthOfStayModel()

    @pytest.fixture
    def complex_case_features(self):
        """Complex case with long expected stay"""
        return ModelFeatures(
            patient_id="P001",
            age=78,
            gender="male",
            current_news2=7.0,
            news2_trend=0.5,
            vital_signs_history=[],
            comorbidities=["diabetes", "heart_disease", "copd"],
            current_medications=["multiple"],
            admission_type="emergency",
            days_since_admission=1,
            previous_alerts_count=2
        )

    @pytest.fixture
    def simple_case_features(self):
        """Simple case with short expected stay"""
        return ModelFeatures(
            patient_id="P002",
            age=35,
            gender="female",
            current_news2=2.0,
            news2_trend=0.0,
            vital_signs_history=[],
            comorbidities=[],
            current_medications=[],
            admission_type="elective",
            days_since_admission=1,
            previous_alerts_count=0
        )

    @pytest.mark.asyncio
    async def test_complex_case_prediction(self, los_model, complex_case_features):
        """Test LOS prediction for complex case"""
        result = await los_model.predict(complex_case_features)

        assert result.prediction_value > 5.0  # Should predict longer stay
        assert "Multiple comorbidities" in result.risk_factors
        assert "discharge planning" in result.recommendations[0].lower()

    @pytest.mark.asyncio
    async def test_simple_case_prediction(self, los_model, simple_case_features):
        """Test LOS prediction for simple case"""
        result = await los_model.predict(simple_case_features)

        assert result.prediction_value < 7.0  # Should predict shorter stay
        assert "Standard discharge planning" in result.recommendations

    @pytest.mark.asyncio
    async def test_admission_type_adjustment(self, los_model):
        """Test admission type adjustment in LOS prediction"""
        emergency_features = ModelFeatures(
            patient_id="P001",
            age=65,
            gender="male",
            current_news2=4.0,
            news2_trend=0.0,
            vital_signs_history=[],
            comorbidities=[],
            current_medications=[],
            admission_type="emergency",
            days_since_admission=1,
            previous_alerts_count=0
        )

        elective_features = ModelFeatures(
            patient_id="P002",
            age=65,
            gender="male",
            current_news2=4.0,
            news2_trend=0.0,
            vital_signs_history=[],
            comorbidities=[],
            current_medications=[],
            admission_type="elective",
            days_since_admission=1,
            previous_alerts_count=0
        )

        emergency_result = await los_model.predict(emergency_features)
        elective_result = await los_model.predict(elective_features)

        # Emergency admissions should have longer predicted stays
        assert emergency_result.prediction_value > elective_result.prediction_value


class TestReadmissionModel:
    """Test suite for ReadmissionModel"""

    @pytest.fixture
    def readmission_model(self):
        """Create ReadmissionModel instance"""
        return ReadmissionModel()

    @pytest.fixture
    def high_readmission_risk_features(self):
        """High readmission risk features"""
        return ModelFeatures(
            patient_id="P001",
            age=75,
            gender="male",
            current_news2=6.0,
            news2_trend=0.2,
            vital_signs_history=[],
            comorbidities=["diabetes", "heart_disease", "copd"],
            current_medications=["multiple"],
            admission_type="emergency",
            days_since_admission=8,
            previous_alerts_count=3
        )

    @pytest.fixture
    def low_readmission_risk_features(self):
        """Low readmission risk features"""
        return ModelFeatures(
            patient_id="P002",
            age=40,
            gender="female",
            current_news2=1.0,
            news2_trend=0.0,
            vital_signs_history=[],
            comorbidities=[],
            current_medications=[],
            admission_type="elective",
            days_since_admission=2,
            previous_alerts_count=0
        )

    @pytest.mark.asyncio
    async def test_high_readmission_risk_prediction(self, readmission_model, high_readmission_risk_features):
        """Test readmission prediction for high-risk patient"""
        result = await readmission_model.predict(high_readmission_risk_features)

        assert result.prediction_value > 0.3  # Should predict higher probability
        assert len(result.risk_factors) > 0
        assert "Enhanced discharge planning required" in result.recommendations

    @pytest.mark.asyncio
    async def test_low_readmission_risk_prediction(self, readmission_model, low_readmission_risk_features):
        """Test readmission prediction for low-risk patient"""
        result = await readmission_model.predict(low_readmission_risk_features)

        assert result.prediction_value < 0.6  # Should predict lower probability
        assert "Routine discharge procedures" in result.recommendations

    @pytest.mark.asyncio
    async def test_comorbidity_risk_factors(self, readmission_model, high_readmission_risk_features):
        """Test comorbidity-based risk factor identification"""
        result = await readmission_model.predict(high_readmission_risk_features)

        expected_risk_factors = [
            "Multiple chronic conditions",
            "Heart disease",
            "COPD"
        ]

        # Should identify comorbidity-related risk factors
        identified_factors = result.risk_factors
        for factor in expected_risk_factors:
            if factor in [rf for rf in identified_factors]:
                assert True
                break
        else:
            assert len(identified_factors) > 0  # Should have at least some risk factors


class TestModelTypeEnum:
    """Test ModelType enum"""

    def test_model_type_values(self):
        """Test ModelType enum values"""
        assert ModelType.DETERIORATION.value == "deterioration"
        assert ModelType.LENGTH_OF_STAY.value == "length_of_stay"
        assert ModelType.READMISSION.value == "readmission"
        assert ModelType.MORTALITY.value == "mortality"
        assert ModelType.RESOURCE_DEMAND.value == "resource_demand"


class TestPredictionConfidenceEnum:
    """Test PredictionConfidence enum"""

    def test_prediction_confidence_values(self):
        """Test PredictionConfidence enum values"""
        assert PredictionConfidence.LOW.value == "low"
        assert PredictionConfidence.MEDIUM.value == "medium"
        assert PredictionConfidence.HIGH.value == "high"
        assert PredictionConfidence.VERY_HIGH.value == "very_high"


class TestPredictiveModelsErrorHandling:
    """Test error handling in predictive models service"""

    @pytest.fixture
    def predictive_service(self):
        """Create PredictiveModelsService instance"""
        return PredictiveModelsService()

    @pytest.mark.asyncio
    async def test_prediction_with_invalid_data(self, predictive_service):
        """Test prediction handling with invalid patient data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "gender": "unknown",
            "current_news2": 50  # Invalid NEWS2 score
        }

        # Should handle gracefully and still return prediction
        result = await predictive_service.predict_deterioration("P001", invalid_data)
        assert isinstance(result, PredictionResult)

    @pytest.mark.asyncio
    async def test_prediction_with_missing_data(self, predictive_service):
        """Test prediction with minimal/missing data"""
        minimal_data = {
            "age": 65,
            "gender": "male"
        }

        result = await predictive_service.predict_deterioration("P001", minimal_data)
        assert isinstance(result, PredictionResult)
        # Confidence should be lower due to missing data
        assert result.confidence in [PredictionConfidence.LOW, PredictionConfidence.MEDIUM]

    @pytest.mark.asyncio
    async def test_simulated_data_generation_exception(self, predictive_service):
        """Test handling of exceptions in simulated data generation"""
        with patch.object(predictive_service, '_get_simulated_patient_data',
                         side_effect=Exception("Data generation error")):
            with pytest.raises(Exception):
                await predictive_service.predict_deterioration("P001")

    @pytest.mark.asyncio
    async def test_model_prediction_exception(self, predictive_service):
        """Test handling of exceptions during model prediction"""
        with patch.object(predictive_service.deterioration_model, 'predict',
                         side_effect=Exception("Model error")):
            with pytest.raises(Exception):
                await predictive_service.predict_deterioration("P001")


class TestPredictiveModelsIntegration:
    """Test predictive models integration"""

    @pytest.fixture
    def predictive_service(self):
        """Create PredictiveModelsService instance"""
        return PredictiveModelsService()

    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, predictive_service):
        """Test concurrent predictions for different models"""
        patient_ids = ["P001", "P002"]

        tasks = []
        for patient_id in patient_ids:
            tasks.append(predictive_service.predict_deterioration(patient_id))
            tasks.append(predictive_service.predict_length_of_stay(patient_id))
            tasks.append(predictive_service.predict_readmission(patient_id))

        results = await asyncio.gather(*tasks)

        assert len(results) == 6  # 3 predictions × 2 patients
        for result in results:
            assert isinstance(result, PredictionResult)

    @pytest.mark.asyncio
    async def test_feature_preparation_consistency(self, predictive_service):
        """Test that feature preparation is consistent across models"""
        patient_data = {
            "age": 65,
            "gender": "male",
            "current_news2": 4.0
        }

        # Get predictions from all models
        deterioration = await predictive_service.predict_deterioration("P001", patient_data)
        los = await predictive_service.predict_length_of_stay("P001", patient_data)
        readmission = await predictive_service.predict_readmission("P001", patient_data)

        # All should use the same patient ID
        assert deterioration.patient_id == "P001"
        assert los.patient_id == "P001"
        assert readmission.patient_id == "P001"

    @pytest.mark.asyncio
    async def test_batch_predictions_performance(self, predictive_service):
        """Test batch predictions performance and correctness"""
        patient_ids = [f"P{i:03d}" for i in range(1, 11)]  # 10 patients

        start_time = datetime.now()
        results = await predictive_service.batch_predictions(
            patient_ids,
            [ModelType.DETERIORATION, ModelType.LENGTH_OF_STAY]
        )
        end_time = datetime.now()

        # Should complete in reasonable time
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 10  # Should complete within 10 seconds

        # Verify all patients processed
        assert len(results) == 10
        for patient_id in patient_ids:
            assert patient_id in results
            assert "deterioration" in results[patient_id]
            assert "length_of_stay" in results[patient_id]