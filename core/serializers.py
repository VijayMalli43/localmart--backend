# LocalMartAI_Project/core/serializers.py

from rest_framework import serializers
from .models import (
    Vendor, Product, CompanyWarehouseProducts, VendorProduct, Review,
    Cart, CartItem, Order, OrderItem,
    Return # Make sure Return model is imported here
)
from django.contrib.auth.models import User # For linking Vendor/Cart/Order to User

# --- Existing Serializers (for Vendor, Product, etc.) ---

# --- User Serializer (for Vendor's linked user and Cart/Order owner) ---
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['username', 'email']

# --- Vendor Serializer ---
class VendorSerializer(serializers.ModelSerializer):
    user_info = UserSerializer(source='user', read_only=True)

    class Meta:
        model = Vendor
        fields = [
            'id', 'user', 'user_info', 'name', 'address', 'reliability_score',
            'phone', 'email', 'is_active_vendor', 'created_at', 'updated_at'
        ]
        read_only_fields = ['reliability_score', 'created_at', 'updated_at', 'user_info']

# --- Product Serializer ---
class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = [
            'id', 'name', 'category', 'description', 'base_price',
            'unit_of_measure', 'image',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

# --- Company Warehouse Products Serializer ---
class CompanyWarehouseProductsSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)

    class Meta:
        model = CompanyWarehouseProducts
        fields = [
            'id', 'product', 'product_name', 'current_quantity', 'price',
            'is_available', 'last_updated'
        ]
        read_only_fields = ['last_updated', 'product_name']

# --- Vendor Product Serializer ---
class VendorProductSerializer(serializers.ModelSerializer):
    vendor_name = serializers.CharField(source='vendor.name', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)

    class Meta:
        model = VendorProduct
        fields = [
            'id', 'vendor', 'vendor_name', 'product', 'product_name',
            'current_quantity', 'price', 'is_available', 'ai_freshness_score', 'last_updated'
        ]
        read_only_fields = ['last_updated', 'vendor_name', 'product_name']

# --- Review Serializer ---
class ReviewSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source='user.username', read_only=True)
    vendor_name = serializers.CharField(source='vendor.name', read_only=True)

    class Meta:
        model = Review
        fields = [
            'id', 'user', 'user_username', 'vendor', 'vendor_name',
            'rating', 'comment', 'ai_sentiment_score', 'created_at'
        ]
        read_only_fields = ['user', 'ai_sentiment_score', 'created_at', 'user_username', 'vendor_name']

# --- New Serializers for Cart & Order Models ---

# --- CartItem Serializer ---
class CartItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)
    unit_of_measure = serializers.CharField(source='product.unit_of_measure', read_only=True)
    base_price = serializers.DecimalField(source='product.base_price', max_digits=10, decimal_places=2, read_only=True)
    product_image = serializers.ImageField(source='product.image', read_only=True)

    class Meta:
        model = CartItem
        fields = [
            'id', 'cart', 'product', 'product_name', 'unit_of_measure',
            'base_price', 'quantity', 'product_image'
        ]
        read_only_fields = ['id', 'product_name', 'unit_of_measure', 'base_price', 'product_image']

# --- Cart Serializer ---
class CartSerializer(serializers.ModelSerializer):
    items = CartItemSerializer(many=True, read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Cart
        fields = [
            'id', 'user', 'user_username', 'items', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user_username', 'items', 'created_at', 'updated_at']

# --- OrderItem Serializer ---
class OrderItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)
    product_image = serializers.ImageField(source='product.image', read_only=True)

    class Meta:
        model = OrderItem
        fields = [
            'id', 'order', 'product', 'product_name', 'quantity',
            'price_at_order', 'source', 'source_vendor', 'product_image'
        ]
        read_only_fields = ['id', 'product_name', 'product_image']

# --- Order Serializer ---
class OrderSerializer(serializers.ModelSerializer):
    items = OrderItemSerializer(many=True, read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)
    fulfilled_by_vendor_name = serializers.CharField(source='fulfilled_by_vendor.name', read_only=True)

    class Meta:
        model = Order
        fields = [
            'id', 'user', 'user_username', 'fulfilled_by_vendor', 'fulfilled_by_vendor_name',
            'order_date', 'total_amount', 'status', 'fulfillment_details', 'items'
        ]
        read_only_fields = [
            'id', 'user_username', 'fulfilled_by_vendor_name', 'order_date',
            'total_amount', 'status', 'fulfillment_details', 'items'
        ]

# --- Return Serializer ---
class ReturnSerializer(serializers.ModelSerializer):
    order_item_product_name = serializers.CharField(source='order_item.product.name', read_only=True)
    order_item_quantity = serializers.DecimalField(source='order_item.quantity', max_digits=10, decimal_places=2, read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Return
        fields = [
            'id', 'order_item', 'order_item_product_name', 'order_item_quantity',
            'user', 'user_username', 'return_quantity', 'return_reason',
            'return_image',
            'ai_return_freshness_prediction', 'ai_prediction_confidence',
            'return_status', 'requested_at', 'processed_at'
        ]
        read_only_fields = [
            'id', 'order_item_product_name', 'order_item_quantity', 'user_username',
            'ai_return_freshness_prediction', 'ai_prediction_confidence',
            'return_status', 'requested_at', 'processed_at'
        ]


# --- Authentication Serializers ---
class RegisterSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name']

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match.")
        return attrs

    def create(self, validated_data):
        # Remove password_confirm as it's not needed for user creation
        validated_data.pop('password_confirm')
        # Create user with encrypted password
        user = User.objects.create_user(**validated_data)
        return user


class LoginSerializer(serializers.Serializer):
    """Serializer for user login"""
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        if username and password:
            from django.contrib.auth import authenticate
            user = authenticate(username=username, password=password)
            if not user:
                raise serializers.ValidationError('Invalid credentials.')
            if not user.is_active:
                raise serializers.ValidationError('User account is disabled.')
            attrs['user'] = user
        else:
            raise serializers.ValidationError('Must include username and password.')
        
        return attrs