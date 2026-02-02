from fastmcp import FastMCP
import psycopg2
import pandas as pd
import warnings
from core.model_core_sync import unpickle
import pickle
import numpy as np
from datetime import datetime
import json

from app.models.mLinearRegression import mLinearRegression
from app.models.mSmesTemp import Smes_temp_model
from app.models.mSmesTempVols import Smes_temp_vols_model
from app.models.mFlashPoint import Smes_fp_model
from app.models.mDNPModel import DNP_model
from app.models.mCloudPoint import Smes_cp_model
from app.models.mCrystKerosene import CrystKerosene
from app.models.mRandomForest import mRandomForest
from app.models.mGradientBoosting import mGradientBoosting
from app.models.mPipeline import mPipeline
from app.models.mMixBonus import Mix_Bonus
from app.models.mMixDupont import Mix_Dupont
from app.models.mMixEthyl import Mix_Ethyl
from app.models.mSmesModel import Smes_model
from app.models.mReservQuality import GetReservQuality
from app.models.mOptimizerModel import OptimizerModel

STATUS_TYPE_ID = 5
MODEL_TYPE_ID = 6
NON_TESTED_MODEL_TYPES = [8, 9, 12]
FILTER_LAYERS = True
DB_ARGS = {
    "dbname": "mnpz_oil_quality", 
    "host": "pgs.dev.local",
    "user": "nk_python", 
    "password": "guylian"
}

if FILTER_LAYERS:
    LAYER1_FILTER = "%_L1_%"
    LAYER2_FILTER = "%_L2_%"
    LAYER3_FILTER = "%_L3_%"
else:
    LAYER1_FILTER = ""
    LAYER2_FILTER = ""
    LAYER3_FILTER = ""

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 2000)
pd.set_option('display.expand_frame_repr', False)

mcp = FastMCP("ModelQualityChecker")

@mcp.tool(description="Поиск моделей без тега статуса")
def find_models_without_status_tag() -> dict:
    """Находит модели без привязанного тега статуса"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH  models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                AND disabled = False
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND name NOT LIKE %(layer1_filter)s
                AND name NOT LIKE %(layer2_filter)s
                AND name NOT LIKE %(layer3_filter)s
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id =  %(status_type_id)s
            ),
            aggr AS (
                SELECT models.id, models.name, statuses.id AS status_id
                FROM models
                LEFT JOIN statuses ON models.id = statuses.parent_id
            )
            SELECT * FROM aggr 
            WHERE status_id IS NULL
        """
        params = {
            "status_type_id": STATUS_TYPE_ID, 
            "model_type_id": MODEL_TYPE_ID, 
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        no_status_tag_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_status_tag_df.empty:
            return {
                
                "message": "У всех моделей есть тег статуса",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_status_tag_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без тега статуса",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без значений статуса в таблице union_values")
def find_models_without_status_values()-> dict:
    """Находит модели без значений статуса в таблице union_values"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND disabled = False
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id =  %(status_type_id)s
            ),
            aggr AS (
                SELECT models.id, models.name, statuses.id AS status_id
                FROM models
                JOIN statuses ON models.id = statuses.parent_id
            ),
            status_values AS (
                SELECT aggr.id, 
                    aggr.name, 
                    aggr.status_id, 
                    MAX(uv.date_value) AS status_date 
                FROM aggr
                LEFT JOIN result.union_values uv ON aggr.status_id = uv.tag_id
                    AND uv.source_type_id =  %(status_type_id)s
                GROUP BY aggr.id, aggr.name, aggr.status_id
            )
            SELECT * FROM status_values WHERE status_date IS NULL
        """
        params = {
            "status_type_id": STATUS_TYPE_ID, 
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        uv_status_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if uv_status_df.empty:
            return {
                
                "message": "У всех моделей есть значения статуса в таблице union_values",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = uv_status_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без значений статуса в таблице union_values",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без значений статуса в таблице union_last_values")
def find_models_without_status_last_values()-> dict:
    """Находит модели без значений статуса в таблице union_last_values"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND disabled = False
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id =  %(status_type_id)s
            ),
            aggr AS (
                SELECT models.id, models.name, statuses.id AS status_id
                FROM models
                JOIN statuses ON models.id = statuses.parent_id
            ),
            status_last_values AS (
                SELECT aggr.id, 
                    aggr.name, 
                    aggr.status_id, 
                    ulv.date_value AS status_date 
                FROM aggr
                LEFT JOIN result.union_last_values ulv ON aggr.status_id = ulv.tag_id
                    AND ulv.source_type_id =  %(status_type_id)s
            )
            SELECT * FROM status_last_values WHERE status_date IS NULL
        """
        params = {
            "status_type_id": STATUS_TYPE_ID, 
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        uv_status_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if uv_status_df.empty:
            return {
                
                "message": "У всех моделей есть значение статуса в таблице union_last_values",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = uv_status_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без значения статуса в таблице union_last_values",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без указанного поля 'lab_mse'")
def find_models_without_lab_mse()-> dict:
    """Находит модели без указанного поля lab_mse"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = False
            AND (attributes->>'lab_mse')::REAL IS NULL
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        no_lab_mse_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_lab_mse_df.empty:
            return {
                
                "message": "У всех моделей указано поле lab_mse",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_lab_mse_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без поля lab_mse",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без указанного поля 'view_diag'")
def find_models_without_view_diag()-> dict:
    """Находит модели без указанного поля view_diag"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = False
            AND (attributes->>'view_diag')::BOOLEAN IS NULL
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        no_view_diag_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_view_diag_df.empty:
            return {
                
                "message": "У всех моделей указано поле view_diag",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_view_diag_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без поля view_diag",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без указанных полей конфигурации ИПК")
def find_models_without_ipk_config()-> dict:
    """Находит модели без полей конфигурации ИПК"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, 
                name,
                ((attributes->'ipk')->>'W_r')::REAL AS w_r,
                ((attributes->'ipk')->>'W_r2')::REAL AS w_r2,
                ((attributes->'ipk')->>'W_mae')::REAL AS w_mae,
                ((attributes->'ipk')->>'ipk_window')::INTEGER AS ipk_window
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
            AND disabled = False
            AND (
                ((attributes->'ipk')->>'W_r')::REAL IS NULL
                OR ((attributes->'ipk')->>'W_r2')::REAL IS NULL
                OR ((attributes->'ipk')->>'W_mae')::REAL IS NULL
                OR ((attributes->'ipk')->>'ipk_window')::INTEGER IS NULL
            )
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        no_ipk_config_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_ipk_config_df.empty:
            return {
                
                "message": "У всех моделей указана конфигурация ИПК",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_ipk_config_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без конфигурации ИПК",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без указанных полей конфигурации дообучения")
def find_models_without_train_config()-> dict:
    """Находит модели без полей конфигурации дообучения"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, 
                name,
                ((attributes->'train_config')->>'ol_mse_mult')::REAL AS ol_mse_mult,
                ((attributes->'train_config')->>'ol_bias_mult')::REAL AS ol_bias_mult,
                ((attributes->'train_config')->>'ol_window')::INTEGER AS ol_window,
                ((attributes->'train_config')->>'cs_mult')::REAL AS cs_mult,
                ((attributes->'train_config')->>'cs_window')::INTEGER AS cs_window
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = False
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
            AND parent_id IS NOT NULL
            AND (
                ((attributes->'train_config')->>'ol_mse_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'ol_bias_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'ol_window')::INTEGER IS NULL
                OR ((attributes->'train_config')->>'cs_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'cs_window')::INTEGER IS NULL
            )
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        no_train_config_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_train_config_df.empty:
            return {
                
                "message": "У всех моделей указана конфигурация дообучения",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_train_config_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без конфигурации дообучения",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без значений за последний час")
def find_models_without_recent_values()-> dict:
    """Находит модели без значений за последний час"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT ut.id, 
                ut.name,
                ulv.date_value AS last_value_date
            FROM dictionary.union_tags ut
            JOIN result.union_last_values ulv ON ut.id = ulv.tag_id
            WHERE ut.source_type_id = %(model_type_id)s
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
            AND disabled = False
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND ulv.date_value < NOW() - interval '1 hour'
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        ulv_hour_old_values = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if ulv_hour_old_values.empty:
            return {
                
                "message": "У всех моделей есть значения в течение последнего часа",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = ulv_hour_old_values.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без значений за последний час",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск отключенных моделей через поле disabled")
def find_disabled_models()-> dict:
    """Находит отключенные модели через поле disabled"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, 
                name,
                disabled
            FROM dictionary.union_tags ut
            WHERE ut.source_type_id = %(model_type_id)s
            AND disabled != False
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        disabled_models = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if disabled_models.empty:
            return {
                
                "message": "Все модели в работе",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = disabled_models.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} отключенных моделей",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без привязки к ЛА")
def find_models_without_parent()-> dict:
    """Находит модели без привязки к ЛА"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, 
                name,
                parent_id
            FROM dictionary.union_tags ut
            WHERE ut.source_type_id = %(model_type_id)s
            AND parent_id IS NULL
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        no_parent_models = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_parent_models.empty:
            return {
                
                "message": "У всех моделей привязаны ЛА",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_parent_models.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без привязки к ЛА",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей, у которых не совпадают единицы измерения с единицами измерения ЛА")
def find_models_with_different_units()-> dict:
    """Находит модели с разными единицами измерения с их ЛА"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT ut.id, 
                ut.name,
                ut.parent_id,
                ut.unit_id AS unit_id,
                parent_ut.unit_id AS parent_unit_id
            FROM dictionary.union_tags ut
            JOIN dictionary.union_tags parent_ut ON ut.parent_id = parent_ut.id
            WHERE ut.source_type_id = %(model_type_id)s
            AND (ut.attributes->>'view_diag')::BOOLEAN=TRUE
            AND ut.unit_id != parent_ut.unit_id
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND ut.name NOT LIKE %(layer1_filter)s
            AND ut.name NOT LIKE %(layer2_filter)s
            AND ut.name NOT LIKE %(layer3_filter)s
            
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        unit_compare_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if unit_compare_df.empty:
            return {
                
                "message": "У всех моделей единицы измерения совпадают с единицами измерения их ЛА",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = unit_compare_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей с разными единицами измерения с их ЛА",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без указанных пределов tech_min и tech_max")
def find_models_without_tech_limits()-> dict:
    """Находит модели без пределов tech_min и tech_max"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT id, 
                name,
                (attributes->>'tech_min')::REAL AS tech_min,
                (attributes->>'tech_max')::REAL AS tech_max
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND (
                (attributes->>'tech_min')::REAL IS NULL
                AND (attributes->>'view_diag')::BOOLEAN=TRUE
                OR (attributes->>'tech_max')::REAL IS NULL
            )
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = False
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        no_limits_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if no_limits_df.empty:
            return {
                
                "message": "У всех моделей указаны пределы tech_min и tech_max",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = no_limits_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без пределов tech_min и tech_max",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей, у которых последнее значение упирается в пределы tech_min или tech_max")
def find_models_with_limited_values()-> dict:
    """Находит модели с значениями упирающимися в пределы"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH models AS (
                SELECT ut.id, 
                    ut.name,
                    (ut.attributes->>'tech_min')::REAL AS tech_min,
                    (ut.attributes->>'tech_max')::REAL AS tech_max,
                    ulv.real_value AS last_value
                FROM dictionary.union_tags ut
                JOIN result.union_last_values ulv ON ut.id = ulv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                WHERE (
                    (attributes->>'tech_min')::REAL IS NOT NULL
                    OR (attributes->>'tech_max')::REAL IS NOT NULL
                )
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND (attributes->>'view_diag')::BOOLEAN=TRUE
                AND disabled = False
            )
            SELECT *, 
                (last_value <= tech_min)::BOOLEAN AS min_bound,
                (last_value >= tech_max)::BOOLEAN AS max_bound 
            FROM models
            WHERE last_value <= tech_min OR last_value >= tech_max
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        limited_values_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        limited_values_df["min_bound"] = limited_values_df["min_bound"].fillna(False)
        limited_values_df["max_bound"] = limited_values_df["max_bound"].fillna(False)
        
        if limited_values_df.empty:
            return {
                
                "message": "У всех моделей значения не упираются в пределы",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = limited_values_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей со значениями упирающимися в пределы",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей, у которых за изменяются значения за последний час")
def find_models_with_frozen_values()-> dict:
    """Находит модели с неизменяющимися значениями за последний час"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH distinct_values AS (
                SELECT DISTINCT ut.id, ut.name, uv.real_value 
                FROM dictionary.union_tags ut
                JOIN result.union_values uv ON ut.id = uv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                WHERE uv.date_value >= NOW() - interval '1 hour'
                AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND (attributes->>'view_diag')::BOOLEAN=TRUE
                AND disabled = False
            ),
            count_values AS (
                SELECT id, name, count(real_value) AS unique_values_count
                FROM distinct_values
                GROUP BY id, name
            )
            SELECT * FROM count_values WHERE unique_values_count <= 1
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES
        }
        frozen_values_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if frozen_values_df.empty:
            return {
                
                "message": "У всех моделей значения изменялись в течение последнего часа",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = frozen_values_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей с неизменяющимися значениями за последний час",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без даты добавления модели")
def find_models_without_evaluation_date()-> dict:
    """Находит модели без даты добавления"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            SELECT ut.id, 
                ut.name,
                (ulv.json_value->>'ecaluation_date')::TIMESTAMP WITH TIME ZONE AS ecaluation_date
            FROM dictionary.union_tags ut
            JOIN result.union_last_values ulv ON ut.id = ulv.tag_id
                AND ut.source_type_id = %(model_type_id)s
            WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ulv.json_value->>'ecaluation_date')::TIMESTAMP WITH TIME ZONE IS NULL
            AND ut.name NOT LIKE %(layer1_filter)s
            AND ut.name NOT LIKE %(layer2_filter)s
            AND ut.name NOT LIKE %(layer3_filter)s
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        ecaluation_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if ecaluation_df.empty:
            return {
                
                "message": "У всех моделей указана дата добавления",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = ecaluation_df.reset_index().to_dict('records')
            return {
                
                "message": f"Найдено {len(models_list)} моделей без даты добавления",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Поиск моделей без метрик ИПК")
def find_models_without_ipk_metrics() -> dict:
    """Находит модели без метрик ИПК"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        query = """
            WITH metrics AS (
                SELECT ut.id,
                    ut.name,
                    ulv.json_value ? 'corr' AS corr,
                    ulv.json_value ? 'det' AS det,
                    ulv.json_value ? 'mae' AS mae,
                    ulv.json_value ? 'Ir' AS Ir,
                    ulv.json_value ? 'Icoef_det' AS Icoef_det,
                    ulv.json_value ? 'Imae' AS Imae,
                    ulv.json_value ? 'ipk' AS ipk,
                    ulv.json_value ? 'sko_la' AS sko_la,
                    -- Проверяем наличие полей в attributes->'ipk'
                    ut.attributes->'ipk'->>'W_r' AS W_r,
                    ut.attributes->'ipk'->>'W_r2' AS W_r2,
                    ut.attributes->'ipk'->>'W_mae' AS W_mae,
                    ut.attributes->'ipk'->>'ipk_window' AS ipk_window,
                    -- Получаем статус модели
                    status_tag.id AS status_tag_id,
                    status_ulv.real_value AS status_value
                FROM dictionary.union_tags ut
                JOIN result.union_last_values ulv ON ut.id = ulv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                -- Присоединяем тег статуса
                LEFT JOIN dictionary.union_tags status_tag ON ut.id = status_tag.parent_id
                    AND status_tag.source_type_id = %(status_type_id)s
                -- Присоединяем последнее значение статуса
                LEFT JOIN result.union_last_values status_ulv ON status_tag.id = status_ulv.tag_id
                    AND status_ulv.source_type_id = %(status_type_id)s
                WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND ut.name NOT LIKE %(layer1_filter)s
                AND ut.name NOT LIKE %(layer2_filter)s
                AND ut.name NOT LIKE %(layer3_filter)s
                AND ut.disabled = False
                AND (ut.attributes->>'view_diag')::BOOLEAN=TRUE
                AND ut.parent_id IS NOT NULL
            )
            SELECT 
                id,
                name,
                corr,
                det,
                mae,
                Ir,
                Icoef_det,
                Imae,
                ipk,
                sko_la,
                W_r,
                W_r2,
                W_mae,
                ipk_window,
                -- Проверяем, равен ли статус 4 (в тесте)
                CASE 
                    WHEN status_value IS NULL THEN NULL
                    WHEN status_value = 4 THEN true
                    ELSE false
                END AS in_test
            FROM metrics
            WHERE (
                NOT corr OR NOT det OR NOT mae
                OR NOT Ir OR NOT Icoef_det OR NOT Imae
                OR NOT ipk OR NOT sko_la
                OR W_r IS NULL OR W_r2 IS NULL OR W_mae IS NULL OR ipk_window IS NULL
            )
            AND ( status_value != 4 )
        """
        params = {
            "model_type_id": MODEL_TYPE_ID,
            "status_type_id": STATUS_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        metrics_df = pd.read_sql(query, conn, params=params, index_col="id").sort_index()
        
        if metrics_df.empty:
            return {
                "message": "У всех моделей указаны метрики ИПК",
                "models_count": 0,
                "models": []
            }
        else:
            models_list = metrics_df.reset_index().to_dict('records')
            return {
                "message": f"Найдено {len(models_list)} моделей без метрик ИПК",
                "models_count": len(models_list),
                "models": models_list
            }

@mcp.tool(description="Проверка на соответствие бинарников и типов моделей")
def check_binary_type_compatibility() -> dict:
    """Проверяет соответствие бинарников и типов моделей"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        model_binaries = {
            0: [mLinearRegression, Smes_temp_model, Smes_temp_vols_model,
                Smes_fp_model, DNP_model, Smes_cp_model,
                CrystKerosene, mRandomForest, mGradientBoosting,
                mPipeline, Mix_Bonus, Mix_Dupont,
                Mix_Ethyl, Smes_model, ],
            1: [Smes_model, ],
            10: [GetReservQuality, ],
            11: [OptimizerModel,],
        }
        query_models = """
            SELECT id, name, (attributes->>'type_model')::INTEGER AS type_model
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = False
            AND (attributes->>'type_model')::INTEGER IS NOT NULL
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
        """
        params_models = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        models_df = pd.read_sql(query_models, conn, params=params_models)
        models_list = []
        for _, row in models_df.iterrows():
            models_list.append({
                "id": int(row["id"]),
                "name": str(row["name"]),
                "type_model": int(row["type_model"])
            })
        
        incorrect_binaries = []
        
        for model_info in models_list:
            model_id = model_info["id"]
            name = model_info["name"]
            type_model = model_info["type_model"]
            query_clusters = """
                WITH clusters AS (
                    SELECT id_model AS model_id,
                        id AS cluster_id,
                        id_cluster AS cluster,
                        RANK() OVER(
                            PARTITION BY id_model, id_cluster 
                            ORDER BY date DESC
                        ) AS cluster_order
                    FROM model.model_cluster_config
                    WHERE id_model = %(model_id)s
                ) 
                SELECT * FROM clusters
                WHERE cluster_order = 1
            """
            clusters_df = pd.read_sql(query_clusters, conn, params={"model_id": model_id})
            clusters_list = []
            for _, row in clusters_df.iterrows():
                clusters_list.append({
                    "cluster_id": int(row["cluster_id"]),
                    "cluster": int(row["cluster"])
                })
            
            for cluster_info in clusters_list:
                cluster = cluster_info["cluster"]
                
                try:
                    query_binary = """
                        WITH binaries AS (
                            SELECT bin_data,
                                row_number() OVER (
                                    PARTITION BY id_model
                                    ORDER BY date desc
                                ) binary_order
                            FROM model.model_binary
                            WHERE id_model = %(model_id)s
                            AND cluster_id = %(cluster)s
                        )
                        SELECT bin_data FROM binaries
                        WHERE binary_order = 1
                        LIMIT 1
                    """
                    binary_data = pd.read_sql(query_binary, conn, params={"model_id": model_id, "cluster": cluster})
                    
                    if binary_data.empty:
                        continue
                    binary_row = binary_data.iloc[0].to_dict()
                    binary = unpickle(binary_row["bin_data"])
                    
                except Exception as e:
                    incorrect_binaries.append({
                        "model_id": model_id,
                        "name": name,
                        "cluster": cluster,
                        "type_model": type_model,
                        "binary_class": None,
                        "error": str(type(e).__name__)
                    })
                    continue
    
                try:
                    model_type_binaries = model_binaries[type_model]
                except KeyError:
                    incorrect_binaries.append({
                        "model_id": model_id,
                        "name": name,
                        "cluster": cluster,
                        "type_model": type_model,
                        "binary_class": type(binary).__name__,
                        "error": f"Неизвестный тип модели {type_model}"
                    })
                    continue
                    
                is_correct_binary = False
                for mtb in model_type_binaries:
                    if isinstance(binary, mtb):
                        is_correct_binary = True
                        break
                        
                if not is_correct_binary:
                    incorrect_binaries.append({
                        "model_id": model_id,
                        "name": name,
                        "cluster": cluster,
                        "type_model": type_model,
                        "binary_class": type(binary).__name__,
                       
                    })
        
        if not incorrect_binaries:
            return {
                "message": "У всех моделей соответствуют бинарники и типы моделей",
                "problems_count": 0,
                "problems": []
            }
        else:
            return {
                "message": f"Найдено {len(incorrect_binaries)} проблем с соответствием бинарников и типов моделей",
                "problems_count": len(incorrect_binaries),
                "problems": incorrect_binaries
            }

@mcp.tool(description="Проверка предикторов, у которых нет значений за последний час")
def check_predictors_without_recent_values()-> dict:
    """Проверяет предикторы без значений за последний час"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        
        query_models = """
            SELECT id, name, (attributes->>'type_model')::INTEGER AS type_model
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = False
            AND (attributes->>'type_model')::INTEGER IS NOT NULL
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
        """
        params_models = {
            "model_type_id": MODEL_TYPE_ID,
            "non_tested_model_types": NON_TESTED_MODEL_TYPES,
            "layer1_filter": LAYER1_FILTER,
            "layer2_filter": LAYER2_FILTER,
            "layer3_filter": LAYER3_FILTER
        }
        models_df = pd.read_sql(query_models, conn, params=params_models, index_col="id").sort_index()
        
        all_outdated_preds = []
        NON_CONTINUOUS_SOURCE_TYPES = [2, 3, 16, 27]
        
        for model_id, model_row in models_df.iterrows():
            
            query_clusters = """
                WITH clusters AS (
                    SELECT id_model AS model_id,
                        id AS cluster_id,
                        id_cluster AS cluster,
                        RANK() OVER(
                            PARTITION BY id_model, id_cluster 
                            ORDER BY date DESC
                        ) AS cluster_order
                    FROM model.model_cluster_config
                    WHERE id_model = %(model_id)s
                ) 
                SELECT * FROM clusters
                WHERE cluster_order = 1
            """
            clusters_df = pd.read_sql(query_clusters, conn, params={"model_id": model_id}).set_index('cluster_id')
            
            for cluster_id, cluster_row in clusters_df.iterrows():
                
                query_predictors = """
                    SELECT mc.id_tag AS tag_id,
                        mc.id_model AS model_id,
                        model.name AS model_name, 
                        ut.name AS tag_name, 
                        ut.source_type_id, 
                        ulv.date_value AS last_date,
                        ulv.date_value < (NOW() - interval '1 hour') OR ulv.date_value IS NULL AS outdated
                    FROM model.model_config mc
                    JOIN dictionary.union_tags ut ON mc.id_tag = ut.id
                    JOIN dictionary.union_tags model ON mc.id_model = model.id
                    LEFT JOIN result.union_last_values ulv ON mc.id_tag = ulv.tag_id
                    WHERE mc.id_model = %(model_id)s
                        AND mc.id_cluster = %(cluster_id)s
                        AND NOT ut.source_type_id = ANY(%(non_continuous_types)s)
                """
                params_predictors = {
                    "model_id": model_id,
                    "cluster_id": cluster_id,
                    "non_continuous_types": NON_CONTINUOUS_SOURCE_TYPES
                }
                pred_df = pd.read_sql(query_predictors, conn, params=params_predictors)
                
                
                outdated_preds = pred_df[pred_df["outdated"]]
                
                for _, pred_row in outdated_preds.iterrows():
                    all_outdated_preds.append({
                        "tag_id": pred_row["tag_id"],
                        "model_id": pred_row["model_id"],
                        "model_name": pred_row["model_name"],
                        "tag_name": pred_row["tag_name"],
                        "source_type_id": pred_row["source_type_id"],
                        "last_date": str(pred_row["last_date"]) if not pd.isna(pred_row["last_date"]) else None,
                        "cluster_id": cluster_id
                    })
        
        if not all_outdated_preds:
            return {
                
                "message": "У всех предикторов есть значения за последний час",
                "outdated_count": 0,
                "outdated_predictors": []
            }
        else:
            return {
                
                "message": f"Найдено {len(all_outdated_preds)} предикторов без значений за последний час",
                "outdated_count": len(all_outdated_preds),
                "outdated_predictors": all_outdated_preds
            }

@mcp.tool(description="Проверка количества предикторов в бинарнике и конфиге")
def check_predictor_count_compatibility() -> dict:
    """Проверяет соответствие количества предикторов в бинарнике и конфигурации"""
    
    with psycopg2.connect(**DB_ARGS) as conn:
        LINEAR_TYPE_MODEL = 0
        
        query_linear = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = False
            AND (attributes->>'type_model')::INTEGER = %(linear_type_model)s
            AND (attributes->>'view_diag')::BOOLEAN=TRUE
        """
        params_linear = {
            "model_type_id": MODEL_TYPE_ID, 
            "linear_type_model": LINEAR_TYPE_MODEL
        }
        linear_df = pd.read_sql(query_linear, conn, params=params_linear)
        linear_models = []
        for _, row in linear_df.iterrows():
            linear_models.append({
                "id": int(row["id"]),
                "name": str(row["name"])
            })
        
        diff_pred_count = []
        
        for model_info in linear_models:
            model_id = model_info["id"]
            name = model_info["name"]
            query_clusters = """
                WITH clusters AS (
                    SELECT id_model AS model_id,
                        id AS cluster_id,
                        id_cluster AS cluster,
                        RANK() OVER(
                            PARTITION BY id_model, id_cluster 
                            ORDER BY date DESC
                        ) AS cluster_order
                    FROM model.model_cluster_config
                    WHERE id_model = %(model_id)s
                ) 
                SELECT * FROM clusters
                WHERE cluster_order = 1
            """
            clusters_df = pd.read_sql(query_clusters, conn, params={"model_id": model_id})
            clusters_list = []
            for _, row in clusters_df.iterrows():
                clusters_list.append({
                    "cluster_id": int(row["cluster_id"]),
                    "cluster": int(row["cluster"])
                })
            
            for cluster_info in clusters_list:
                cluster = cluster_info["cluster"]
                cluster_id = cluster_info["cluster_id"]
                
                try:
                    query_binary = """
                        WITH binaries AS (
                            SELECT bin_data,
                                row_number() OVER (
                                    PARTITION BY id_model
                                    ORDER BY date desc
                                ) binary_order
                            FROM model.model_binary
                            WHERE id_model = %(model_id)s
                            AND cluster_id = %(cluster)s
                        )
                        SELECT bin_data FROM binaries
                        WHERE binary_order = 1
                        LIMIT 1
                    """
                    binary_data = pd.read_sql(query_binary, conn, params={"model_id": model_id, "cluster": cluster})
                    
                    if binary_data.empty:
                        continue
                    
                    binary_row = binary_data.iloc[0].to_dict()
                    binary = unpickle(binary_row["bin_data"])
                    
                except Exception:
                    continue
                    
                if isinstance(binary, mLinearRegression):
                    query_count = """
                        SELECT COUNT(*) AS cnt
                        FROM model.model_config
                        WHERE id_model = %(model_id)s
                            AND id_cluster = %(cluster_id)s
                    """
                    count_data = pd.read_sql(query_count, conn, params={"model_id": model_id, "cluster_id": cluster_id})
                    
                    config_pred_count = 0
                    if not count_data.empty:
                        config_pred_count = int(count_data.iloc[0]["cnt"])

                    binary_pred_count = int(len(np.ravel(binary.coef_).tolist()))
                    
                    if config_pred_count != binary_pred_count:
                        diff_pred_count.append({
                            "model_id": model_id,
                            "cluster": cluster,
                            "name": name,
                            "config_pred_count": config_pred_count,
                            "binary_pred_count": binary_pred_count
                        })
        
        if not diff_pred_count:
            return {
                "message": "У всех линейных моделей совпадает число предикторов в бинарнике и конфигурации",
                "mismatch_count": 0,
                "mismatches": []
            }
        else:
            return {
                "message": f"Найдено {len(diff_pred_count)} линейных моделей с несовпадающим числом предикторов",
                "mismatch_count": len(diff_pred_count),
                "mismatches": diff_pred_count
            }

if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8000, path="/mcp")