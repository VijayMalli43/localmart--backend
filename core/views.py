# LocalMartAI_Project/core/views.py

from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db import transaction
from decimal import Decimal # Used for precise quantity/price calculations
import random # Used in dummy AI model call
import os # Needed for os.path.join

# --- AI Specific Imports and LOCAL Model Loading ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# --- Dynamic ML Engine Import ---
from .ml_engine import ml_engine

# Define the local path to your trained sentiment model
# This path is relative to the 'core' app directory
# Ensure your 'trained_sentiment_model' folder (from Colab download) is inside 'core/ai_models/sentiment_model/'
LOCAL_SENTIMENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ai_models', 'sentiment_model')

try:
    print(f"--- Loading LOCAL AI Sentiment Model from {LOCAL_SENTIMENT_MODEL_PATH}... ---")
    # Load the tokenizer and model from your local saved path
    tokenizer_sentiment = AutoTokenizer.from_pretrained(LOCAL_SENTIMENT_MODEL_PATH)
    # num_labels should match your training (2 for binary '1' and '2' from the .bz2 dataset)
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(LOCAL_SENTIMENT_MODEL_PATH, num_labels=2)

    # Create a pipeline from the loaded local model for inference
    sentiment_classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    print("--- LOCAL AI Sentiment Model Loaded. ---")

    # Check if a GPU is available for inference (if you want to use it on your Django server, requires CUDA setup)
    if torch.cuda.is_available():
        model_sentiment.to('cuda')
        print("--- Using GPU for AI inference in Django. ---")
    else:
        print("--- Using CPU for AI inference in Django. ---")

except Exception as e:
    sentiment_classifier = None # Set to None so code can run without AI features
    print(f"ERROR: Could not load LOCAL AI Sentiment Model from {LOCAL_SENTIMENT_MODEL_PATH}: {e}")
    print("Sentiment analysis features will be unavailable.")


# --- Import ALL your models ---
from .models import (
    Vendor, Product, CompanyWarehouseProducts, VendorProduct, Review,
    Cart, CartItem, Order, OrderItem,
    Return
)
# --- Import Django's built-in User model and Token for authentication ---
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.permissions import IsAuthenticated


# --- Import ALL your serializers ---
from .serializers import (
    UserSerializer,
    VendorSerializer, ProductSerializer, CompanyWarehouseProductsSerializer,
    VendorProductSerializer, ReviewSerializer,
    CartSerializer, CartItemSerializer, OrderSerializer, OrderItemSerializer,
    ReturnSerializer,
    RegisterSerializer, LoginSerializer
)


# --- Existing ViewSets (Basic CRUD for Admin/Data Management) ---
class VendorViewSet(viewsets.ModelViewSet):
    queryset = Vendor.objects.all().order_by('-reliability_score')
    serializer_class = VendorSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all().order_by('name')
    serializer_class = ProductSerializer

class CompanyWarehouseProductsViewSet(viewsets.ModelViewSet):
    queryset = CompanyWarehouseProducts.objects.all().order_by('product__name')
    serializer_class = CompanyWarehouseProductsSerializer

class VendorProductViewSet(viewsets.ModelViewSet):
    queryset = VendorProduct.objects.all().order_by('vendor__name', 'product__name')
    serializer_class = VendorProductSerializer

# --- Review ViewSet (UPDATED: Added AI integration for sentiment) ---
class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.all().order_by('-created_at')
    serializer_class = ReviewSerializer
    permission_classes = [IsAuthenticated]  # Require authentication for reviews

    # Override perform_create to integrate AI sentiment and update vendor reliability
    def perform_create(self, serializer):
        # BUSINESS LOGIC: Validate user has received delivered order from this vendor
        vendor = serializer.validated_data.get('vendor')
        user = self.request.user
        
        # Check if user has any delivered orders from this vendor
        delivered_orders = Order.objects.filter(
            user=user,
            fulfilled_by_vendor=vendor,
            status='Delivered'
        )
        
        if not delivered_orders.exists():
            from rest_framework.exceptions import ValidationError
            raise ValidationError({
                'detail': f'You cannot review {vendor.name} until you have received a delivered order from them. Please confirm delivery first.'
            })
        
        comment = serializer.validated_data.get('comment')
        ai_sentiment_score = None
        
        if sentiment_classifier and comment: # Only run AI if model loaded and comment exists
            try:
                results = sentiment_classifier(comment)
                if results:
                    label = results[0]['label'] # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                    score = results[0]['score'] # Confidence score
                    # Map sentiment to a 0-1 score for reliability calculation
                    # The Amazon dataset's labels ('1' and '2') need to be correctly mapped to POSITIVE/NEGATIVE/NEUTRAL.
                    # Assuming '1' maps to negative and '2' maps to positive (common convention for FastText sentiment)
                    # You might need to adjust this mapping if your dataset labels mean something else
                    if label == 'LABEL_1': # Often means negative in some datasets
                        ai_sentiment_score = Decimal(str(0.5 - (score / 2))) # Maps to 0.0-0.5
                    elif label == 'LABEL_2': # Often means positive in some datasets
                        ai_sentiment_score = Decimal(str(0.5 + (score / 2))) # Maps to 0.5-1.0
                    else: # Fallback for unexpected labels
                        ai_sentiment_score = Decimal('0.50') # Neutral

                    print(f"AI Sentiment for Review: '{comment[:50]}...' -> Label: {label} (Confidence: {score:.2f}) -> Score: {ai_sentiment_score:.2f}")
            except Exception as e:
                print(f"Error during AI sentiment prediction for review: {e}")
                ai_sentiment_score = None

        # Save the review with the AI sentiment score and automatically set the user
        review = serializer.save(user=self.request.user, ai_sentiment_score=ai_sentiment_score)

        # Update Vendor's reliability_score based on the new review's sentiment
        vendor = review.vendor
        if ai_sentiment_score is not None:
            # Weighted average update for reliability: e.g., 20% review rating, 30% AI sentiment, 50% old score
            # This ensures reliability score moves dynamically based on new feedback
            new_reliability = (review.rating / Decimal('5.00') * Decimal('0.20')) + \
                              (ai_sentiment_score * Decimal('0.30')) + \
                              (vendor.reliability_score * Decimal('0.50'))
            
            # Cap the score between 0.00 and 1.00
            vendor.reliability_score = min(Decimal('1.00'), max(Decimal('0.00'), new_reliability))
            vendor.save()
            print(f"Updated {vendor.name}'s reliability to: {vendor.reliability_score:.2f}")
        else:
            print("AI sentiment not available, Vendor reliability not updated from this review.")


# --- New ViewSets for Cart & Order (Basic CRUD) ---
class CartViewSet(viewsets.ModelViewSet):
    queryset = Cart.objects.all()
    serializer_class = CartSerializer

class CartItemViewSet(viewsets.ModelViewSet):
    queryset = CartItem.objects.all()
    serializer_class = CartItemSerializer

class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer

class OrderItemViewSet(viewsets.ModelViewSet):
    queryset = OrderItem.objects.all()
    serializer_class = OrderItemSerializer


# --- Custom API Views for Cart & Fulfillment Logic ---

# Helper function to get or create a cart for a user (or a dummy user for now)
def get_or_create_user_cart(user):
    if not user.is_authenticated:
        # Use get_or_create to avoid duplicate user creation
        user_obj, created = User.objects.get_or_create(
            username='temp_cart_user',
            defaults={'password': 'temp_password'}
        )
        user = user_obj

    cart, created = Cart.objects.get_or_create(user=user)
    return cart

# --- Add Item to Cart API ---
class AddToCartAPIView(APIView):
    """
    Adds a product to the user's cart or updates its quantity.
    Expects: {'product_id': <int>, 'quantity': <decimal>}
    """
    def post(self, request, *args, **kwargs):
        product_id = request.data.get('product_id')
        quantity = request.data.get('quantity')

        if not all([product_id, quantity]):
            return Response({'error': 'Product ID and quantity are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            product = Product.objects.get(id=product_id)
            quantity = Decimal(quantity)
        except Product.DoesNotExist:
            return Response({'error': 'Product not found.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': f'Invalid quantity or product ID: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        cart = get_or_create_user_cart(request.user)

        cart_item, created = CartItem.objects.get_or_create(cart=cart, product=product, defaults={'quantity': quantity})

        if not created:
            cart_item.quantity += quantity
            cart_item.save()

        serializer = CartItemSerializer(cart_item)
        return Response(serializer.data, status=status.HTTP_200_OK if not created else status.HTTP_201_CREATED)

# --- Remove Item from Cart API ---
class RemoveFromCartAPIView(APIView):
    """
    Removes a product from the cart or decreases its quantity.
    Expects: {'product_id': <int>, 'quantity': <decimal> (optional, if removing partial)}
    """
    def post(self, request, *args, **kwargs):
        product_id = request.data.get('product_id')
        quantity_to_remove = request.data.get('quantity')

        if not product_id:
            return Response({'error': 'Product ID is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            product = Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            return Response({'error': 'Product not found.'}, status=status.HTTP_404_NOT_FOUND)

        cart = get_or_create_user_cart(request.user)

        try:
            cart_item = CartItem.objects.get(cart=cart, product=product)
        except CartItem.DoesNotExist:
            return Response({'error': 'Product not in cart.'}, status=status.HTTP_404_NOT_FOUND)

        if quantity_to_remove is not None:
            quantity_to_remove = Decimal(quantity_to_remove)
            if cart_item.quantity <= quantity_to_remove:
                cart_item.delete()
            else:
                cart_item.quantity -= quantity_to_remove
                cart_item.save()
        else:
            cart_item.delete()

        return Response({'status': 'Item removed/quantity updated.'}, status=status.HTTP_200_OK)


# --- Calculate Fulfillment API (Core Logic) ---
class CalculateFulfillmentAPIView(APIView):
    """
    Calculates the best fulfillment option for the current user's cart.
    Logic: Warehouse first, then single best vendor using ML logic.
    """
    def get(self, request, *args, **kwargs):
        user_cart = get_or_create_user_cart(request.user)
        cart_items = user_cart.items.all()

        if not cart_items.exists():
            return Response({'message': 'Cart is empty.'}, status=status.HTTP_200_OK)

        fulfillment_plan = {
            'recommended_source': 'None',
            'details': {},
            'total_cart_value': Decimal('0.00'),
            'unfulfilled_items': []
        }

        items_to_fulfill = {item.product: item.quantity for item in cart_items}
        remaining_items = dict(items_to_fulfill)

        # 1. --- Try to fulfill from Company Warehouse First ---
        warehouse_fulfillment = {}
        warehouse_total_value = Decimal('0.00')

        for product, requested_qty in items_to_fulfill.items():
            try:
                warehouse_stock = CompanyWarehouseProducts.objects.get(product=product, is_available=True)
                available_qty = warehouse_stock.current_quantity

                if available_qty >= requested_qty:
                    warehouse_fulfillment[product.name] = {
                        'quantity': requested_qty,
                        'price_per_unit': warehouse_stock.price,
                        'source': 'Warehouse',
                        'total_item_value': requested_qty * warehouse_stock.price
                    }
                    remaining_items.pop(product)
                    warehouse_total_value += requested_qty * warehouse_stock.price
                elif available_qty > 0:
                    warehouse_fulfillment[product.name] = {
                        'quantity': available_qty,
                        'price_per_unit': warehouse_stock.price,
                        'source': 'Warehouse (Partial)',
                        'total_item_value': available_qty * warehouse_stock.price
                    }
                    remaining_items[product] -= available_qty
                    warehouse_total_value += available_qty * warehouse_stock.price

            except CompanyWarehouseProducts.DoesNotExist:
                pass

        if not remaining_items:
            fulfillment_plan['recommended_source'] = 'Company Warehouse'
            fulfillment_plan['details'] = warehouse_fulfillment
            fulfillment_plan['total_cart_value'] = warehouse_total_value
            return Response(fulfillment_plan, status=status.HTTP_200_OK)

        # 2. --- If not fully fulfilled by warehouse, find best single Vendor ---
        potential_vendors = {}

        for product, requested_qty in remaining_items.items():
            vendor_products = VendorProduct.objects.filter(
                product=product,
                is_available=True,
                current_quantity__gt=0
            ).select_related('vendor')

            for vp in vendor_products:
                vendor_obj = vp.vendor
                if vendor_obj.id not in potential_vendors:
                    potential_vendors[vendor_obj.id] = {
                        'vendor_obj': vendor_obj,
                        'items_fulfilled_count': 0,
                        'total_items_in_cart': len(cart_items),
                        'total_value_if_fulfilled': Decimal('0.00'),
                        'items_details': {}
                    }
                
                potential_vendors[vendor_obj.id]['items_fulfilled_count'] += 1
                # Use available quantity vs requested quantity for partial fulfillment
                available_qty = min(vp.current_quantity, requested_qty)
                potential_vendors[vendor_obj.id]['items_details'][product.name] = {
                    'quantity': available_qty,
                    'price_per_unit': vp.price,
                    'source': vendor_obj.name,
                    'total_item_value': available_qty * vp.price
                }
                potential_vendors[vendor_obj.id]['total_value_if_fulfilled'] += available_qty * vp.price

        best_vendor_recommendation = None
        max_fulfillment_score = -1

        product_ids = [product.id for product in remaining_items.keys()]
        ml_predictions = ml_engine.get_best_vendors_for_products(product_ids)
        
        for vendor_id, data in potential_vendors.items():
            # Allow vendors that can fulfill at least some items (not requiring full fulfillment)
            if data['items_fulfilled_count'] > 0:
                # Traditional reliability score
                traditional_score = float(data['vendor_obj'].reliability_score)
                
                # Get ML-based vendor performance prediction
                ml_score = 50.0  # Default ML score
                for product_id in product_ids:
                    vendor_ml_scores = ml_predictions.get(product_id, {})
                    if vendor_id in vendor_ml_scores:
                        ml_score = max(ml_score, float(vendor_ml_scores[vendor_id]))
                
                # Get AI freshness score for this vendor's products (Future enhancement)
                freshness_score = self._get_vendor_avg_freshness_score(data['vendor_obj'], remaining_items.keys())
                
                # Combine scores: 50% traditional reliability, 25% ML prediction, 25% freshness
                combined_score = (traditional_score * 0.5) + (ml_score * 0.25) + (freshness_score * 0.25)
                
                # Update vendor data with ML insights
                data['traditional_score'] = traditional_score
                data['ml_score'] = ml_score
                data['freshness_score'] = freshness_score
                data['combined_score'] = combined_score

                if best_vendor_recommendation is None or combined_score > max_fulfillment_score:
                    max_fulfillment_score = combined_score
                    best_vendor_recommendation = data

        if best_vendor_recommendation:
            fulfillment_plan['recommended_source'] = 'Single Vendor (ML-Enhanced)'
            fulfillment_plan['details'] = best_vendor_recommendation['items_details']
            fulfillment_plan['total_cart_value'] = float(fulfillment_plan['total_cart_value']) + float(best_vendor_recommendation['total_value_if_fulfilled'])
            fulfillment_plan['recommended_vendor_id'] = best_vendor_recommendation['vendor_obj'].id
            fulfillment_plan['recommended_vendor_name'] = best_vendor_recommendation['vendor_obj'].name
            fulfillment_plan['ml_insights'] = {
                'traditional_score': round(best_vendor_recommendation.get('traditional_score', 0), 2),
                'ml_prediction_score': round(best_vendor_recommendation.get('ml_score', 0), 2),
                'freshness_score': round(best_vendor_recommendation.get('freshness_score', 0), 2),
                'combined_score': round(best_vendor_recommendation.get('combined_score', 0), 2),
                'training_status': ml_engine.get_training_status()
            }
        else:
            fulfillment_plan['recommended_source'] = 'Mixed / Partial'
            fulfillment_plan['details'] = warehouse_fulfillment
            fulfillment_plan['total_cart_value'] = warehouse_total_value
            for product, qty in remaining_items.items():
                fulfillment_plan['unfulfilled_items'].append({
                    'product_id': product.id,
                    'product_name': product.name,
                    'quantity_needed': qty,
                    'status': 'Not fully fulfilled by single vendor'
                })

        return Response(fulfillment_plan, status=status.HTTP_200_OK)

    def _get_vendor_avg_freshness_score(self, vendor, products):
        """
        Calculate average freshness score for vendor across requested products
        Future scope: Enhanced with real-time freshness detection
        """
        try:
            vendor_products = VendorProduct.objects.filter(
                vendor=vendor,
                product__in=products,
                ai_freshness_score__isnull=False
            )
            
            if vendor_products.exists():
                from django.db import models
                avg_freshness = vendor_products.aggregate(
                    avg_score=models.Avg('ai_freshness_score')
                )['avg_score']
                return float(avg_freshness) if avg_freshness else 75.0
            else:
                return 75.0  # Default freshness score (0.75 on 0-1 scale)
        except Exception as e:
            print(f"Error calculating vendor freshness: {e}")
            return 75.0


# --- Place Order API ---
class PlaceOrderAPIView(APIView):
    """
    Places an order based on the current user's cart and selected fulfillment plan.
    Decrements stock from the respective warehouse/vendor.
    Expects: {'user_id': <int>, 'fulfilled_by_vendor_id': <int> (optional, if vendor fulfilled)}
             Optionally, a 'fulfillment_details' JSON object from CalculateFulfillmentAPIView result.
    """
    @transaction.atomic
    def post(self, request, *args, **kwargs):
        fulfilled_by_vendor_id = request.data.get('fulfilled_by_vendor_id')
        client_fulfillment_details = request.data.get('fulfillment_details')
        delivery_address = request.data.get('delivery_address')
        delivery_notes = request.data.get('delivery_notes', '')

        # Use authenticated user from token
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required.'}, status=status.HTTP_401_UNAUTHORIZED)

        user = request.user

        cart = get_or_create_user_cart(user)
        cart_items = cart.items.all()

        if not cart_items.exists():
            return Response({'error': 'Cannot place order: Cart is empty.'}, status=status.HTTP_400_BAD_REQUEST)
        
        fulfilled_by_vendor = None
        if fulfilled_by_vendor_id:
            try:
                fulfilled_by_vendor = Vendor.objects.get(id=fulfilled_by_vendor_id)
            except Vendor.DoesNotExist:
                return Response({'error': 'Specified fulfillment vendor not found.'}, status=status.HTTP_404_NOT_FOUND)

        total_amount = Decimal('0.00')
        order_items_to_create = []
        
        items_to_decrement_stock = {}

        if client_fulfillment_details and 'details' in client_fulfillment_details:
            for product_name, item_detail in client_fulfillment_details['details'].items():
                try:
                    product_obj = Product.objects.get(name=product_name)
                    qty_ordered = Decimal(item_detail['quantity'])
                    price_at_order = Decimal(item_detail['price_per_unit'])
                    source_type = 'Warehouse' if item_detail['source'] == 'Warehouse' else 'Vendor'
                    source_vendor_obj = None

                    if source_type == 'Vendor':
                        source_vendor_obj = Vendor.objects.filter(name=item_detail['source']).first()
                        if not source_vendor_obj:
                            source_vendor_obj = fulfilled_by_vendor

                    if source_type == 'Warehouse':
                        current_stock_obj = CompanyWarehouseProducts.objects.select_for_update().get(product=product_obj, is_available=True)
                        if current_stock_obj.current_quantity < qty_ordered:
                            raise ValueError(f"Not enough stock in warehouse for {product_name}.")
                        items_to_decrement_stock[(product_obj.id, 'Warehouse', None)] = qty_ordered
                    else:
                        current_stock_obj = VendorProduct.objects.select_for_update().get(vendor=source_vendor_obj, product=product_obj, is_available=True)
                        if current_stock_obj.current_quantity < qty_ordered:
                            raise ValueError(f"Not enough stock at {source_vendor_obj.name} for {product_name}.")
                        items_to_decrement_stock[(product_obj.id, 'Vendor', source_vendor_obj.id)] = qty_ordered

                    order_items_to_create.append({
                        'product': product_obj,
                        'quantity': qty_ordered,
                        'price_at_order': price_at_order,
                        'source': source_type,
                        'source_vendor': source_vendor_obj
                    })
                    total_amount += qty_ordered * price_at_order

                except Product.DoesNotExist:
                    return Response({'error': f"Product '{product_name}' from fulfillment details not found."}, status=status.HTTP_400_BAD_REQUEST)
                except (CompanyWarehouseProducts.DoesNotExist, VendorProduct.DoesNotExist):
                    return Response({'error': f"Stock for {product_name} not found or unavailable."}, status=status.HTTP_404_NOT_FOUND)
                except ValueError as e:
                    return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'Fulfillment details are required to place an order.'}, status=status.HTTP_400_BAD_REQUEST)

        order = Order.objects.create(
            user=user,
            fulfilled_by_vendor=fulfilled_by_vendor,
            total_amount=total_amount,
            status='Pending',
            fulfillment_details=client_fulfillment_details
        )

        for item_data in order_items_to_create:
            OrderItem.objects.create(order=order, **item_data)
            
            product_id = item_data['product'].id
            qty_to_decrement = item_data['quantity']
            source_type = item_data['source']
            source_vendor_id = item_data['source_vendor'].id if item_data['source_vendor'] else None

            if source_type == 'Warehouse':
                warehouse_stock = CompanyWarehouseProducts.objects.get(product_id=product_id)
                warehouse_stock.current_quantity -= qty_to_decrement
                warehouse_stock.save()
            else:
                vendor_stock = VendorProduct.objects.get(vendor_id=source_vendor_id, product_id=product_id)
                vendor_stock.current_quantity -= qty_to_decrement
                vendor_stock.save()

        cart_items.delete()

        serializer = OrderSerializer(order)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


# --- Return Item API ---
class ReturnItemAPIView(APIView):
    """
    Process a return request for an order item.
    Expects: {'order_item_id': <int>, 'return_quantity': <decimal>, 'return_reason': <str>, 'return_image': <file>}
    """
    def post(self, request, *args, **kwargs):
        order_item_id = request.data.get('order_item_id')
        return_quantity = request.data.get('return_quantity')
        return_reason = request.data.get('return_reason')
        return_image = request.FILES.get('return_image')

        if not all([order_item_id, return_quantity, return_reason]):
            return Response({'error': 'Order item ID, return quantity, and return reason are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            order_item = OrderItem.objects.get(id=order_item_id)
            return_quantity = Decimal(return_quantity)
        except OrderItem.DoesNotExist:
            return Response({'error': 'Order item not found.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': f'Invalid return quantity: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        if return_quantity > order_item.quantity:
            return Response({'error': 'Return quantity cannot exceed ordered quantity.'}, status=status.HTTP_400_BAD_REQUEST)

        # Create return request
        return_request = Return.objects.create(
            order_item=order_item,
            user=order_item.order.user,
            return_quantity=return_quantity,
            return_reason=return_reason,
            return_image=return_image,
            return_status='Pending'
        )

        # AI Freshness Prediction for Return Items
        ai_freshness_prediction = None
        ai_confidence = None
        
        # Future scope: Real AI freshness detection using computer vision
        # For now, use intelligent dummy logic based on return reason
        if return_reason:
            reason_lower = return_reason.lower()
            if any(word in reason_lower for word in ['rotten', 'spoiled', 'expired', 'bad', 'moldy']):
                ai_freshness_prediction = 'Rotten'
                ai_confidence = Decimal(str(random.uniform(0.85, 0.98)))
            elif any(word in reason_lower for word in ['damaged', 'broken', 'crushed', 'torn']):
                ai_freshness_prediction = 'Damaged'
                ai_confidence = Decimal(str(random.uniform(0.75, 0.90)))
            elif any(word in reason_lower for word in ['fresh', 'good', 'quality', 'fine']):
                ai_freshness_prediction = 'Fresh'
                ai_confidence = Decimal(str(random.uniform(0.60, 0.85)))
            else:
                ai_freshness_prediction = 'Slightly Used'
                ai_confidence = Decimal(str(random.uniform(0.50, 0.75)))
        else:
            # Default fallback
            ai_freshness_prediction = random.choice(['Fresh', 'Slightly Used', 'Damaged'])
            ai_confidence = Decimal(str(random.uniform(0.50, 0.95)))

        return_request.ai_return_freshness_prediction = ai_freshness_prediction
        return_request.ai_prediction_confidence = ai_confidence
        return_request.save()

        # Update vendor freshness score based on return quality (Future AI enhancement)
        self._update_vendor_freshness_score(order_item, ai_freshness_prediction, ai_confidence)

        serializer = ReturnSerializer(return_request)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def _update_vendor_freshness_score(self, order_item, freshness_prediction, confidence):
        """
        Update vendor's product freshness score based on return quality
        Future scope: This will be enhanced with real computer vision AI
        """
        try:
            if order_item.source == 'Vendor' and order_item.source_vendor:
                vendor_product = VendorProduct.objects.get(
                    vendor=order_item.source_vendor,
                    product=order_item.product
                )
                
                # Calculate freshness impact based on prediction
                freshness_impact = Decimal('0.00')
                if freshness_prediction == 'Rotten':
                    freshness_impact = Decimal('-0.20')  # Significant negative impact
                elif freshness_prediction == 'Damaged':
                    freshness_impact = Decimal('-0.10')  # Moderate negative impact
                elif freshness_prediction == 'Slightly Used':
                    freshness_impact = Decimal('-0.05')  # Minor negative impact
                elif freshness_prediction == 'Fresh':
                    freshness_impact = Decimal('0.02')   # Small positive impact (good return)
                
                # Apply confidence weighting
                weighted_impact = freshness_impact * (confidence or Decimal('0.50'))
                
                # Update vendor product freshness score
                current_score = vendor_product.ai_freshness_score or Decimal('0.75')  # Default fresh score
                new_score = current_score + weighted_impact
                
                # Cap between 0.00 and 1.00
                vendor_product.ai_freshness_score = min(Decimal('1.00'), max(Decimal('0.00'), new_score))
                vendor_product.save()
                
                print(f"Updated {vendor_product.vendor.name}'s {vendor_product.product.name} freshness: {current_score:.2f} -> {vendor_product.ai_freshness_score:.2f} (Return: {freshness_prediction})")
                
        except VendorProduct.DoesNotExist:
            pass  # Warehouse products don't have freshness tracking
        except Exception as e:
            print(f"Error updating vendor freshness score: {e}")


# --- Delivery Confirmation API ---
class ConfirmDeliveryAPIView(APIView):
    """
    Confirm delivery of an order (simulates user clicking 'Confirm Delivery' button)
    Order ID is passed as URL parameter: /api/orders/confirm-delivery/{order_id}/
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        order_id = kwargs.get('order_id')
        
        if not order_id:
            return Response({'error': 'Order ID is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            order = Order.objects.get(id=order_id, user=request.user)
        except Order.DoesNotExist:
            return Response({'error': 'Order not found or not yours.'}, status=status.HTTP_404_NOT_FOUND)
        
        # Only allow confirmation if order is 'Out for Delivery'
        if order.status != 'Out for Delivery':
            return Response({
                'error': f'Cannot confirm delivery. Order status is: {order.status}. Must be "Out for Delivery".'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update order status to Delivered
        order.status = 'Delivered'
        order.save()
        
        return Response({
            'message': 'Delivery confirmed successfully! You can now review this vendor.',
            'order_id': order.id,
            'status': order.status,
            'can_review': True
        }, status=status.HTTP_200_OK)


# --- Update Order Status API (Admin/Vendor) ---
class UpdateOrderStatusAPIView(APIView):
    """
    Admin/Vendor API to update order status (simulates delivery progress)
    Order ID is passed as URL parameter: /api/orders/update-status/{order_id}/
    Expects: {'status': <str>}
    """
    def post(self, request, *args, **kwargs):
        order_id = kwargs.get('order_id')
        new_status = request.data.get('status')
        
        if not order_id:
            return Response({'error': 'Order ID is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not new_status:
            return Response({'error': 'Status is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate status
        valid_statuses = ['Pending', 'Processing', 'Out for Delivery', 'Delivered', 'Cancelled']
        if new_status not in valid_statuses:
            return Response({'error': f'Invalid status. Must be one of: {valid_statuses}'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            order = Order.objects.get(id=order_id)
        except Order.DoesNotExist:
            return Response({'error': 'Order not found.'}, status=status.HTTP_404_NOT_FOUND)
        
        # Update order status
        order.status = new_status
        order.save()
        
        return Response({
            'message': f'Order status updated to: {new_status}',
            'order_id': order.id,
            'status': order.status
        }, status=status.HTTP_200_OK)


# --- User Authentication APIs ---
class RegisterUserAPIView(APIView):
    """
    Register a new user account.
    Expects: {'username': <str>, 'email': <str>, 'password': <str>, 'password_confirm': <str>, 'first_name': <str>, 'last_name': <str>}
    """
    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # Create auth token for the new user
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'user': UserSerializer(user).data,
                'token': token.key,
                'message': 'User registered successfully.'
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginUserAPIView(APIView):
    """
    Login a user and return authentication token.
    Expects: {'username': <str>, 'password': <str>}
    """
    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'user': UserSerializer(user).data,
                'token': token.key,
                'message': 'Login successful.'
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutUserAPIView(APIView):
    """
    Logout a user by deleting their authentication token.
    Requires authentication token in header.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            # Get the token from the request
            token = request.auth
            if token:
                token.delete()
                return Response({'message': 'Successfully logged out.'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'No active session found.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Logout failed: {e}'}, status=status.HTTP_400_BAD_REQUEST)


class UserProfileAPIView(APIView):
    """
    Get current authenticated user's profile information.
    Requires authentication token in header.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'message': 'User profile retrieved successfully.'
        }, status=status.HTTP_200_OK)
