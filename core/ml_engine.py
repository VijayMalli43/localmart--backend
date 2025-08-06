# LocalMartAI_Project/core/ml_engine.py
# Dynamic ML Recommendation Engine

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os
from decimal import Decimal
from django.db.models import Avg, Count, Sum
from datetime import datetime, timedelta
from .models import Vendor, Product, Review, Order, OrderItem, Return, VendorProduct, CompanyWarehouseProducts

class DynamicMLEngine:
    """
    Dynamic ML Engine that learns from live database data
    Features: Vendor recommendation, demand forecasting, quality prediction
    """
    
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'ai_models', 'dynamic_ml')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Model files
        self.vendor_model_path = os.path.join(self.model_path, 'vendor_recommender.pkl')
        self.scaler_path = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load existing models or initialize new ones
        self.vendor_model = self._load_model(self.vendor_model_path)
        self.scaler = self._load_scaler(self.scaler_path)
        
        print("🤖 Dynamic ML Engine initialized")
    
    def _load_model(self, path):
        """Load existing model or return None"""
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"✅ Loaded existing model from {path}")
                return model
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
        return None
    
    def _load_scaler(self, path):
        """Load existing scaler or create new one"""
        try:
            if os.path.exists(path):
                scaler = joblib.load(path)
                print(f"✅ Loaded existing scaler from {path}")
                return scaler
        except Exception as e:
            print(f"⚠️ Could not load scaler: {e}")
        return StandardScaler()
    
    def extract_features_from_database(self):
        """Extract ML features from current database state"""
        print("📊 Extracting features from database...")
        
        features = []
        targets = []
        
        # Get all vendors with their performance metrics
        vendors = Vendor.objects.all()
        
        for vendor in vendors:
            # Review-based features
            reviews = Review.objects.filter(vendor=vendor)
            if reviews.exists():
                avg_rating = reviews.aggregate(Avg('rating'))['rating__avg'] or 0
                review_count = reviews.count()
                avg_sentiment = reviews.aggregate(Avg('ai_sentiment_score'))['ai_sentiment_score__avg'] or 0.5
            else:
                avg_rating = 2.5  # Default neutral rating
                review_count = 0
                avg_sentiment = 0.5
            
            # Order-based features (when you have orders)
            orders = Order.objects.filter(fulfilled_by_vendor=vendor)
            if orders.exists():
                total_orders = orders.count()
                avg_order_value = float(orders.aggregate(Avg('total_amount'))['total_amount__avg'] or 0)
                
                # Return rate calculation
                order_items = OrderItem.objects.filter(order__fulfilled_by_vendor=vendor)
                returns = Return.objects.filter(order_item__order__fulfilled_by_vendor=vendor)
                return_rate = (returns.count() / max(order_items.count(), 1)) * 100
            else:
                total_orders = 0
                avg_order_value = 0.0
                return_rate = 0.0
            
            # Product availability features
            vendor_products = VendorProduct.objects.filter(vendor=vendor, is_available=True)
            if vendor_products.exists():
                avg_stock = float(vendor_products.aggregate(Avg('current_quantity'))['current_quantity__avg'] or 0)
                product_variety = vendor_products.count()
                avg_price = float(vendor_products.aggregate(Avg('price'))['price__avg'] or 0)
            else:
                avg_stock = 0.0
                product_variety = 0
                avg_price = 0.0
            
            # Business metrics
            days_since_created = (datetime.now().date() - vendor.created_at.date()).days if hasattr(vendor, 'created_at') else 30
            
            # Feature vector for this vendor
            feature_vector = [
                float(avg_rating),           # Average rating (0-5)
                float(review_count),         # Number of reviews
                float(avg_sentiment),        # AI sentiment score (0-1)
                float(total_orders),         # Total orders fulfilled
                float(avg_order_value),      # Average order value
                float(return_rate),          # Return rate percentage
                float(avg_stock),            # Average stock quantity
                float(product_variety),      # Number of different products
                float(avg_price),            # Average product price
                float(days_since_created),   # Business longevity
                float(vendor.reliability_score)  # Current reliability score
            ]
            
            features.append(feature_vector)
            
            # Target: Future reliability prediction based on current score + trend
            # This will be what we want to predict for new vendors
            target_reliability = float(vendor.reliability_score)
            targets.append(target_reliability)
        
        print(f"📈 Extracted {len(features)} vendor feature vectors")
        return np.array(features), np.array(targets)
    
    def train_vendor_recommender(self):
        """Train the vendor recommendation model with current database data"""
        print("🔄 Training vendor recommendation model...")
        
        try:
            # Extract features from database
            X, y = self.extract_features_from_database()
            
            if len(X) < 2:
                print("⚠️ Not enough data to train. Need at least 2 vendors with data.")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model (using Random Forest for interpretability)
            self.vendor_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=2
            )
            
            self.vendor_model.fit(X_scaled, y)
            
            # Save models
            joblib.dump(self.vendor_model, self.vendor_model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            print("✅ Vendor recommendation model trained and saved!")
            
            # Print feature importance
            feature_names = [
                'avg_rating', 'review_count', 'avg_sentiment', 'total_orders',
                'avg_order_value', 'return_rate', 'avg_stock', 'product_variety',
                'avg_price', 'days_since_created', 'current_reliability'
            ]
            
            importance = self.vendor_model.feature_importances_
            for name, imp in zip(feature_names, importance):
                print(f"📊 {name}: {imp:.3f}")
            
            # Store training metadata
            from datetime import datetime
            self.last_training_time = datetime.now()
            self.training_samples = len(X)
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def predict_vendor_performance(self, vendor_id):
        """Predict future performance of a specific vendor"""
        if self.vendor_model is None:
            print("⚠️ No trained model available. Training first...")
            if not self.train_vendor_recommender():
                return None
        
        try:
            vendor = Vendor.objects.get(id=vendor_id)
            
            # Extract features for this vendor (same as in training)
            reviews = Review.objects.filter(vendor=vendor)
            if reviews.exists():
                avg_rating = reviews.aggregate(Avg('rating'))['rating__avg'] or 0
                review_count = reviews.count()
                avg_sentiment = reviews.aggregate(Avg('ai_sentiment_score'))['ai_sentiment_score__avg'] or 0.5
            else:
                avg_rating = 2.5
                review_count = 0
                avg_sentiment = 0.5
            
            orders = Order.objects.filter(fulfilled_by_vendor=vendor)
            if orders.exists():
                total_orders = orders.count()
                avg_order_value = float(orders.aggregate(Avg('total_amount'))['total_amount__avg'] or 0)
                
                order_items = OrderItem.objects.filter(order__fulfilled_by_vendor=vendor)
                returns = Return.objects.filter(order_item__order__fulfilled_by_vendor=vendor)
                return_rate = (returns.count() / max(order_items.count(), 1)) * 100
            else:
                total_orders = 0
                avg_order_value = 0.0
                return_rate = 0.0
            
            vendor_products = VendorProduct.objects.filter(vendor=vendor, is_available=True)
            if vendor_products.exists():
                avg_stock = float(vendor_products.aggregate(Avg('current_quantity'))['current_quantity__avg'] or 0)
                product_variety = vendor_products.count()
                avg_price = float(vendor_products.aggregate(Avg('price'))['price__avg'] or 0)
            else:
                avg_stock = 0.0
                product_variety = 0
                avg_price = 0.0
            
            days_since_created = (datetime.now().date() - vendor.created_at.date()).days if hasattr(vendor, 'created_at') else 30
            
            # Create feature vector
            features = np.array([[
                float(avg_rating), float(review_count), float(avg_sentiment),
                float(total_orders), float(avg_order_value), float(return_rate),
                float(avg_stock), float(product_variety), float(avg_price),
                float(days_since_created), float(vendor.reliability_score)
            ]])
            
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            prediction = self.vendor_model.predict(features_scaled)[0]
            
            return {
                'vendor_id': vendor_id,
                'vendor_name': vendor.name,
                'current_reliability': float(vendor.reliability_score),
                'predicted_reliability': float(prediction),
                'improvement_potential': float(prediction) - float(vendor.reliability_score),
                'confidence': 0.85  # You can add confidence intervals later
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def get_best_vendors_for_products(self, product_ids, quantity_needed=1):
        """Get ML-ranked vendor recommendations for specific products"""
        print(f"🎯 Finding best vendors for products: {product_ids}")
        
        recommendations = {}
        
        for product_id in product_ids:
            try:
                product = Product.objects.get(id=product_id)
                
                # Get vendors who have this product
                vendor_products = VendorProduct.objects.filter(
                    product=product,
                    is_available=True,
                    current_quantity__gte=quantity_needed
                ).select_related('vendor')
                
                vendor_scores = {}
                
                for vp in vendor_products:
                    # Get ML prediction for this vendor
                    ml_prediction = self.predict_vendor_performance(vp.vendor.id)
                    
                    if ml_prediction:
                        score = ml_prediction['predicted_reliability']
                    else:
                        score = float(vp.vendor.reliability_score)
                    
                    vendor_scores[vp.vendor.id] = score
                
                recommendations[product_id] = vendor_scores
                
            except Product.DoesNotExist:
                print(f"⚠️ Product {product_id} not found")
                continue
        
        return recommendations
    
    def retrain_on_new_data(self):
        """Automatically retrain when new data is available"""
        print("🔄 Retraining model with latest database data...")
        return self.train_vendor_recommender()
    
    def get_training_status(self):
        """Get the current training status of the ML models"""
        status = {
            'vendor_model_trained': self.vendor_model is not None,
            'last_training_time': getattr(self, 'last_training_time', None),
            'total_training_samples': getattr(self, 'training_samples', 0),
            'model_version': '1.0'
        }
        
        if hasattr(self, 'last_training_time') and self.last_training_time:
            status['last_training_time'] = self.last_training_time.isoformat()
        
        return status

# Global ML engine instance
ml_engine = DynamicMLEngine()
