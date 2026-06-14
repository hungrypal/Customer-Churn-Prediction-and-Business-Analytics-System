import pandas as pd
import mysql.connector
from mysql.connector import Error
from backend.config import DB_CONFIG, DATA_DIR
from typing import Optional, List, Dict
import json

class DatabaseManager:
    """Manage MySQL database operations"""
    
    def __init__(self):
        self.connection = None
    
    def create_connection(self) -> bool:
        """Create database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database']
            )
            if self.connection.is_connected():
                print("Connected to MySQL database")
                return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
        return False
    
    def create_tables(self):
        """Create necessary tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_id VARCHAR(50) UNIQUE NOT NULL,
                tenure INT,
                monthly_charges DECIMAL(10,2),
                total_charges DECIMAL(10,2),
                contract_type VARCHAR(50),
                payment_method VARCHAR(50),
                internet_service VARCHAR(50),
                churn INT,
                prediction_score DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Predictions log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_id VARCHAR(50),
                model_used VARCHAR(50),
                prediction INT,
                probability DECIMAL(5,4),
                features JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        cursor.close()
    
    def insert_customer(self, customer_data: Dict) -> bool:
        """Insert or update customer"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO customers 
                (customer_id, tenure, monthly_charges, total_charges, 
                 contract_type, payment_method, internet_service, churn, prediction_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                tenure = VALUES(tenure),
                monthly_charges = VALUES(monthly_charges),
                total_charges = VALUES(total_charges),
                updated_at = CURRENT_TIMESTAMP
            """, (
                customer_data['customer_id'],
                customer_data.get('tenure', 0),
                customer_data.get('monthly_charges', 0),
                customer_data.get('total_charges', 0),
                customer_data.get('contract_type', ''),
                customer_data.get('payment_method', ''),
                customer_data.get('internet_service', ''),
                customer_data.get('churn', 0),
                customer_data.get('prediction_score', 0)
            ))
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error inserting customer: {e}")
            return False
        finally:
            cursor.close()
    
    def log_prediction(self, customer_id: str, model: str, prediction: int, 
                      probability: float, features: Dict) -> bool:
        """Log prediction"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO prediction_logs 
                (customer_id, model_used, prediction, probability, features)
                VALUES (%s, %s, %s, %s, %s)
            """, (customer_id, model, prediction, probability, json.dumps(features)))
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error logging prediction: {e}")
            return False
        finally:
            cursor.close()
    
    def get_statistics(self) -> Dict:
        """Get dashboard statistics"""
        if not self.connection:
            return {}
        
        cursor = self.connection.cursor(dictionary=True)
        try:
            stats = {}
            
            # Total customers
            cursor.execute("SELECT COUNT(*) as count FROM customers")
            stats['total_customers'] = cursor.fetchone()['count']
            
            # Churned customers
            cursor.execute("SELECT COUNT(*) as count FROM customers WHERE churn = 1")
            stats['churned_customers'] = cursor.fetchone()['count']
            
            # Churn rate
            if stats['total_customers'] > 0:
                stats['churn_rate'] = round(
                    (stats['churned_customers'] / stats['total_customers']) * 100, 2
                )
            else:
                stats['churn_rate'] = 0
            
            # Recent predictions
            cursor.execute("""
                SELECT COUNT(*) as count FROM prediction_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            stats['recent_predictions'] = cursor.fetchone()['count']
            
            return stats
        except Error as e:
            print(f"Error getting statistics: {e}")
            return {}
        finally:
            cursor.close()
    
    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")