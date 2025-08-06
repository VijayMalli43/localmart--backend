# Register your models here.
from django.contrib import admin
from .models import (
    Vendor, Product, CompanyWarehouseProducts, VendorProduct, 
    Review, Cart, CartItem, Order, OrderItem, Return
)

# Basic admin registrations for easy data management
@admin.register(Vendor)
class VendorAdmin(admin.ModelAdmin):
    list_display = ['name', 'reliability_score', 'is_active_vendor', 'created_at']
    search_fields = ['name', 'email']
    list_filter = ['is_active_vendor', 'reliability_score']

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'base_price', 'unit_of_measure']
    search_fields = ['name', 'category']
    list_filter = ['category']

@admin.register(VendorProduct)
class VendorProductAdmin(admin.ModelAdmin):
    list_display = ['vendor', 'product', 'price', 'current_quantity', 'is_available']
    search_fields = ['vendor__name', 'product__name']
    list_filter = ['is_available', 'vendor']

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status', 'total_amount', 'order_date']
    search_fields = ['user__username', 'id']
    list_filter = ['status', 'order_date']

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ['user', 'vendor', 'rating', 'ai_sentiment_score', 'created_at']
    search_fields = ['user__username', 'vendor__name']
    list_filter = ['rating', 'created_at']

# Register remaining models with default admin
admin.site.register(CompanyWarehouseProducts)
admin.site.register(Cart)
admin.site.register(CartItem)
admin.site.register(OrderItem)
admin.site.register(Return)
