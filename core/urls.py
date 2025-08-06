# LocalMartAI_Project/core/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    # ViewSets
    VendorViewSet, ProductViewSet, CompanyWarehouseProductsViewSet,
    VendorProductViewSet, ReviewViewSet,
    CartViewSet, CartItemViewSet, OrderViewSet, OrderItemViewSet,

    # Custom API Views
    AddToCartAPIView, RemoveFromCartAPIView, CalculateFulfillmentAPIView,
    PlaceOrderAPIView,
    ReturnItemAPIView,
    ConfirmDeliveryAPIView, UpdateOrderStatusAPIView,

    # New User Authentication APIs
    RegisterUserAPIView, LoginUserAPIView, LogoutUserAPIView, UserProfileAPIView
)

# Default Router for standard CRUD ViewSets
router = DefaultRouter()
router.register(r'vendors', VendorViewSet)
router.register(r'products', ProductViewSet)
router.register(r'warehouse-products', CompanyWarehouseProductsViewSet)
router.register(r'vendor-products', VendorProductViewSet)
router.register(r'reviews', ReviewViewSet)
router.register(r'carts', CartViewSet)
router.register(r'cart-items', CartItemViewSet)
router.register(r'orders', OrderViewSet)
router.register(r'order-items', OrderItemViewSet)

# Custom URL patterns for specific business logic
urlpatterns = [
    path('', include(router.urls)), # Includes all routes from the DefaultRouter

    # Custom Cart & Order API Endpoints
    path('cart/add/', AddToCartAPIView.as_view(), name='cart-add'),
    path('cart/remove/', RemoveFromCartAPIView.as_view(), name='cart-remove'),
    path('cart/calculate-fulfillment/', CalculateFulfillmentAPIView.as_view(), name='cart-calculate-fulfillment'),
    path('cart/place-order/', PlaceOrderAPIView.as_view(), name='cart-place-order'),

    # Returns API Endpoint
    path('returns/process/', ReturnItemAPIView.as_view(), name='process-return'),

    # Delivery Confirmation API Endpoints
    path('orders/confirm-delivery/<int:order_id>/', ConfirmDeliveryAPIView.as_view(), name='confirm-delivery'),
    path('orders/update-status/<int:order_id>/', UpdateOrderStatusAPIView.as_view(), name='update-order-status'),

    # New User Authentication API Endpoints
    path('auth/register/', RegisterUserAPIView.as_view(), name='register'),
    path('auth/login/', LoginUserAPIView.as_view(), name='login'),
    path('auth/logout/', LogoutUserAPIView.as_view(), name='logout'),
    path('auth/user/', UserProfileAPIView.as_view(), name='user-profile'),
]