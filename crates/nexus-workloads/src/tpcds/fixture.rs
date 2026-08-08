use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::WorkloadResult;

pub fn write_tiny_tpcds_parquet_dataset(root: &Path) -> WorkloadResult<()> {
    fs::create_dir_all(root)?;
    write_call_center(root)?;
    write_catalog_page(root)?;
    write_catalog_returns(root)?;
    write_catalog_sales(root)?;
    write_customer(root)?;
    write_customer_address(root)?;
    write_customer_demographics(root)?;
    write_date_dim(root)?;
    write_household_demographics(root)?;
    write_income_band(root)?;
    write_inventory(root)?;
    write_item(root)?;
    write_promotion(root)?;
    write_reason(root)?;
    write_ship_mode(root)?;
    write_store(root)?;
    write_store_returns(root)?;
    write_store_sales(root)?;
    write_time_dim(root)?;
    write_warehouse(root)?;
    write_web_page(root)?;
    write_web_returns(root)?;
    write_web_sales(root)?;
    write_web_site(root)?;
    Ok(())
}

fn write_call_center(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "call_center",
        schema(vec![
            field("cc_call_center_sk", DataType::Int64),
            field("cc_call_center_id", DataType::Utf8),
            field("cc_rec_start_date", DataType::Date32),
            field("cc_rec_end_date", DataType::Date32),
            field("cc_closed_date_sk", DataType::Int64),
            field("cc_open_date_sk", DataType::Int64),
            field("cc_name", DataType::Utf8),
            field("cc_class", DataType::Utf8),
            field("cc_employees", DataType::Int64),
            field("cc_sq_ft", DataType::Int64),
            field("cc_hours", DataType::Utf8),
            field("cc_manager", DataType::Utf8),
            field("cc_mkt_id", DataType::Int64),
            field("cc_mkt_class", DataType::Utf8),
            field("cc_mkt_desc", DataType::Utf8),
            field("cc_market_manager", DataType::Utf8),
            field("cc_division", DataType::Int64),
            field("cc_division_name", DataType::Utf8),
            field("cc_company", DataType::Int64),
            field("cc_company_name", DataType::Utf8),
            field("cc_street_number", DataType::Utf8),
            field("cc_street_name", DataType::Utf8),
            field("cc_street_type", DataType::Utf8),
            field("cc_suite_number", DataType::Utf8),
            field("cc_city", DataType::Utf8),
            field("cc_county", DataType::Utf8),
            field("cc_state", DataType::Utf8),
            field("cc_zip", DataType::Utf8),
            field("cc_country", DataType::Utf8),
            field("cc_gmt_offset", DataType::Decimal128(5, 2)),
            field("cc_tax_percentage", DataType::Decimal128(5, 2)),
        ]),
        vec![
            i64s([0, 1]),
            strings(["cc_call_center_id_a", "cc_call_center_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["cc_name_a", "cc_name_b"]),
            strings(["cc_class_a", "cc_class_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["cc_hours_a", "cc_hours_b"]),
            strings(["cc_manager_a", "cc_manager_b"]),
            i64s([0, 1]),
            strings(["cc_mkt_class_a", "cc_mkt_class_b"]),
            strings(["cc_mkt_desc_a", "cc_mkt_desc_b"]),
            strings(["cc_market_manager_a", "cc_market_manager_b"]),
            i64s([0, 1]),
            strings(["cc_division_name_a", "cc_division_name_b"]),
            i64s([0, 1]),
            strings(["cc_company_name_a", "cc_company_name_b"]),
            strings(["cc_street_number_a", "cc_street_number_b"]),
            strings(["cc_street_name_a", "cc_street_name_b"]),
            strings(["cc_street_type_a", "cc_street_type_b"]),
            strings(["cc_suite_number_a", "cc_suite_number_b"]),
            strings(["cc_city_a", "cc_city_b"]),
            strings(["cc_county_a", "cc_county_b"]),
            strings(["cc_state_a", "cc_state_b"]),
            strings(["cc_zip_a", "cc_zip_b"]),
            strings(["cc_country_a", "cc_country_b"]),
            decimals_5_2([0, 100]),
            decimals_5_2([0, 100]),
        ],
    )
}

fn write_catalog_page(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "catalog_page",
        schema(vec![
            field("cp_catalog_page_sk", DataType::Int64),
            field("cp_catalog_page_id", DataType::Utf8),
            field("cp_start_date_sk", DataType::Int64),
            field("cp_end_date_sk", DataType::Int64),
            field("cp_department", DataType::Utf8),
            field("cp_catalog_number", DataType::Int64),
            field("cp_catalog_page_number", DataType::Int64),
            field("cp_description", DataType::Utf8),
            field("cp_type", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["cp_catalog_page_id_a", "cp_catalog_page_id_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["cp_department_a", "cp_department_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["cp_description_a", "cp_description_b"]),
            strings(["cp_type_a", "cp_type_b"]),
        ],
    )
}

fn write_catalog_returns(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "catalog_returns",
        schema(vec![
            field("cr_returned_date_sk", DataType::Int64),
            field("cr_returned_time_sk", DataType::Int64),
            field("cr_item_sk", DataType::Int64),
            field("cr_refunded_customer_sk", DataType::Int64),
            field("cr_refunded_cdemo_sk", DataType::Int64),
            field("cr_refunded_hdemo_sk", DataType::Int64),
            field("cr_refunded_addr_sk", DataType::Int64),
            field("cr_returning_customer_sk", DataType::Int64),
            field("cr_returning_cdemo_sk", DataType::Int64),
            field("cr_returning_hdemo_sk", DataType::Int64),
            field("cr_returning_addr_sk", DataType::Int64),
            field("cr_call_center_sk", DataType::Int64),
            field("cr_catalog_page_sk", DataType::Int64),
            field("cr_ship_mode_sk", DataType::Int64),
            field("cr_warehouse_sk", DataType::Int64),
            field("cr_reason_sk", DataType::Int64),
            field("cr_order_number", DataType::Int64),
            field("cr_return_quantity", DataType::Int64),
            field("cr_return_amount", DataType::Decimal128(7, 2)),
            field("cr_return_tax", DataType::Decimal128(7, 2)),
            field("cr_return_amt_inc_tax", DataType::Decimal128(7, 2)),
            field("cr_fee", DataType::Decimal128(7, 2)),
            field("cr_return_ship_cost", DataType::Decimal128(7, 2)),
            field("cr_refunded_cash", DataType::Decimal128(7, 2)),
            field("cr_reversed_charge", DataType::Decimal128(7, 2)),
            field("cr_store_credit", DataType::Decimal128(7, 2)),
            field("cr_net_loss", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_catalog_sales(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "catalog_sales",
        schema(vec![
            field("cs_sold_date_sk", DataType::Int64),
            field("cs_sold_time_sk", DataType::Int64),
            field("cs_ship_date_sk", DataType::Int64),
            field("cs_bill_customer_sk", DataType::Int64),
            field("cs_bill_cdemo_sk", DataType::Int64),
            field("cs_bill_hdemo_sk", DataType::Int64),
            field("cs_bill_addr_sk", DataType::Int64),
            field("cs_ship_customer_sk", DataType::Int64),
            field("cs_ship_cdemo_sk", DataType::Int64),
            field("cs_ship_hdemo_sk", DataType::Int64),
            field("cs_ship_addr_sk", DataType::Int64),
            field("cs_call_center_sk", DataType::Int64),
            field("cs_catalog_page_sk", DataType::Int64),
            field("cs_ship_mode_sk", DataType::Int64),
            field("cs_warehouse_sk", DataType::Int64),
            field("cs_item_sk", DataType::Int64),
            field("cs_promo_sk", DataType::Int64),
            field("cs_order_number", DataType::Int64),
            field("cs_quantity", DataType::Int64),
            field("cs_wholesale_cost", DataType::Decimal128(7, 2)),
            field("cs_list_price", DataType::Decimal128(7, 2)),
            field("cs_sales_price", DataType::Decimal128(7, 2)),
            field("cs_ext_discount_amt", DataType::Decimal128(7, 2)),
            field("cs_ext_sales_price", DataType::Decimal128(7, 2)),
            field("cs_ext_wholesale_cost", DataType::Decimal128(7, 2)),
            field("cs_ext_list_price", DataType::Decimal128(7, 2)),
            field("cs_ext_tax", DataType::Decimal128(7, 2)),
            field("cs_coupon_amt", DataType::Decimal128(7, 2)),
            field("cs_ext_ship_cost", DataType::Decimal128(7, 2)),
            field("cs_net_paid", DataType::Decimal128(7, 2)),
            field("cs_net_paid_inc_tax", DataType::Decimal128(7, 2)),
            field("cs_net_paid_inc_ship", DataType::Decimal128(7, 2)),
            field("cs_net_paid_inc_ship_tax", DataType::Decimal128(7, 2)),
            field("cs_net_profit", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_customer(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "customer",
        schema(vec![
            field("c_customer_sk", DataType::Int64),
            field("c_customer_id", DataType::Utf8),
            field("c_current_cdemo_sk", DataType::Int64),
            field("c_current_hdemo_sk", DataType::Int64),
            field("c_current_addr_sk", DataType::Int64),
            field("c_first_shipto_date_sk", DataType::Int64),
            field("c_first_sales_date_sk", DataType::Int64),
            field("c_salutation", DataType::Utf8),
            field("c_first_name", DataType::Utf8),
            field("c_last_name", DataType::Utf8),
            field("c_preferred_cust_flag", DataType::Utf8),
            field("c_birth_day", DataType::Int64),
            field("c_birth_month", DataType::Int64),
            field("c_birth_year", DataType::Int64),
            field("c_birth_country", DataType::Utf8),
            field("c_login", DataType::Utf8),
            field("c_email_address", DataType::Utf8),
            field("c_last_review_date_sk", DataType::Int32),
        ]),
        vec![
            i64s([0, 1]),
            strings(["c_customer_id_a", "c_customer_id_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["c_salutation_a", "c_salutation_b"]),
            strings(["c_first_name_a", "c_first_name_b"]),
            strings(["c_last_name_a", "c_last_name_b"]),
            strings(["c_preferred_cust_flag_a", "c_preferred_cust_flag_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["c_birth_country_a", "c_birth_country_b"]),
            strings(["c_login_a", "c_login_b"]),
            strings(["c_email_address_a", "c_email_address_b"]),
            i32s([0, 1]),
        ],
    )
}

fn write_customer_address(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "customer_address",
        schema(vec![
            field("ca_address_sk", DataType::Int64),
            field("ca_address_id", DataType::Utf8),
            field("ca_street_number", DataType::Utf8),
            field("ca_street_name", DataType::Utf8),
            field("ca_street_type", DataType::Utf8),
            field("ca_suite_number", DataType::Utf8),
            field("ca_city", DataType::Utf8),
            field("ca_county", DataType::Utf8),
            field("ca_state", DataType::Utf8),
            field("ca_zip", DataType::Utf8),
            field("ca_country", DataType::Utf8),
            field("ca_gmt_offset", DataType::Decimal128(5, 2)),
            field("ca_location_type", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["ca_address_id_a", "ca_address_id_b"]),
            strings(["ca_street_number_a", "ca_street_number_b"]),
            strings(["ca_street_name_a", "ca_street_name_b"]),
            strings(["ca_street_type_a", "ca_street_type_b"]),
            strings(["ca_suite_number_a", "ca_suite_number_b"]),
            strings(["ca_city_a", "ca_city_b"]),
            strings(["ca_county_a", "ca_county_b"]),
            strings(["ca_state_a", "ca_state_b"]),
            strings(["ca_zip_a", "ca_zip_b"]),
            strings(["ca_country_a", "ca_country_b"]),
            decimals_5_2([0, 100]),
            strings(["ca_location_type_a", "ca_location_type_b"]),
        ],
    )
}

fn write_customer_demographics(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "customer_demographics",
        schema(vec![
            field("cd_demo_sk", DataType::Int64),
            field("cd_gender", DataType::Utf8),
            field("cd_marital_status", DataType::Utf8),
            field("cd_education_status", DataType::Utf8),
            field("cd_purchase_estimate", DataType::Int64),
            field("cd_credit_rating", DataType::Utf8),
            field("cd_dep_count", DataType::Int64),
            field("cd_dep_employed_count", DataType::Int64),
            field("cd_dep_college_count", DataType::Int32),
        ]),
        vec![
            i64s([0, 1]),
            strings(["cd_gender_a", "cd_gender_b"]),
            strings(["cd_marital_status_a", "cd_marital_status_b"]),
            strings(["cd_education_status_a", "cd_education_status_b"]),
            i64s([0, 1]),
            strings(["cd_credit_rating_a", "cd_credit_rating_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i32s([0, 1]),
        ],
    )
}

fn write_date_dim(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "date_dim",
        schema(vec![
            field("d_date_sk", DataType::Int64),
            field("d_date_id", DataType::Utf8),
            field("d_date", DataType::Date32),
            field("d_month_seq", DataType::Int64),
            field("d_week_seq", DataType::Int64),
            field("d_quarter_seq", DataType::Int64),
            field("d_year", DataType::Int64),
            field("d_dow", DataType::Int64),
            field("d_moy", DataType::Int64),
            field("d_dom", DataType::Int64),
            field("d_qoy", DataType::Int64),
            field("d_fy_year", DataType::Int64),
            field("d_fy_quarter_seq", DataType::Int64),
            field("d_fy_week_seq", DataType::Int64),
            field("d_day_name", DataType::Utf8),
            field("d_quarter_name", DataType::Utf8),
            field("d_holiday", DataType::Utf8),
            field("d_weekend", DataType::Utf8),
            field("d_following_holiday", DataType::Utf8),
            field("d_first_dom", DataType::Int64),
            field("d_last_dom", DataType::Int64),
            field("d_same_day_ly", DataType::Int64),
            field("d_same_day_lq", DataType::Int64),
            field("d_current_day", DataType::Utf8),
            field("d_current_week", DataType::Utf8),
            field("d_current_month", DataType::Utf8),
            field("d_current_quarter", DataType::Utf8),
            field("d_current_year", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["d_date_id_a", "d_date_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["d_day_name_a", "d_day_name_b"]),
            strings(["d_quarter_name_a", "d_quarter_name_b"]),
            strings(["d_holiday_a", "d_holiday_b"]),
            strings(["d_weekend_a", "d_weekend_b"]),
            strings(["d_following_holiday_a", "d_following_holiday_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["d_current_day_a", "d_current_day_b"]),
            strings(["d_current_week_a", "d_current_week_b"]),
            strings(["d_current_month_a", "d_current_month_b"]),
            strings(["d_current_quarter_a", "d_current_quarter_b"]),
            strings(["d_current_year_a", "d_current_year_b"]),
        ],
    )
}

fn write_household_demographics(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "household_demographics",
        schema(vec![
            field("hd_demo_sk", DataType::Int64),
            field("hd_income_band_sk", DataType::Int64),
            field("hd_buy_potential", DataType::Utf8),
            field("hd_dep_count", DataType::Int64),
            field("hd_vehicle_count", DataType::Int32),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["hd_buy_potential_a", "hd_buy_potential_b"]),
            i64s([0, 1]),
            i32s([0, 1]),
        ],
    )
}

fn write_income_band(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "income_band",
        schema(vec![
            field("ib_income_band_sk", DataType::Int64),
            field("ib_lower_bound", DataType::Int64),
            field("ib_upper_bound", DataType::Int32),
        ]),
        vec![i64s([0, 1]), i64s([0, 1]), i32s([0, 1])],
    )
}

fn write_inventory(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "inventory",
        schema(vec![
            field("inv_date_sk", DataType::Int64),
            field("inv_item_sk", DataType::Int64),
            field("inv_warehouse_sk", DataType::Int64),
            field("inv_quantity_on_hand", DataType::Int32),
        ]),
        vec![i64s([0, 1]), i64s([0, 1]), i64s([0, 1]), i32s([0, 1])],
    )
}

fn write_item(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "item",
        schema(vec![
            field("i_item_sk", DataType::Int64),
            field("i_item_id", DataType::Utf8),
            field("i_rec_start_date", DataType::Date32),
            field("i_rec_end_date", DataType::Date32),
            field("i_item_desc", DataType::Utf8),
            field("i_current_price", DataType::Decimal128(7, 2)),
            field("i_wholesale_cost", DataType::Decimal128(7, 2)),
            field("i_brand_id", DataType::Int64),
            field("i_brand", DataType::Utf8),
            field("i_class_id", DataType::Int64),
            field("i_class", DataType::Utf8),
            field("i_category_id", DataType::Int64),
            field("i_category", DataType::Utf8),
            field("i_manufact_id", DataType::Int64),
            field("i_manufact", DataType::Utf8),
            field("i_size", DataType::Utf8),
            field("i_formulation", DataType::Utf8),
            field("i_color", DataType::Utf8),
            field("i_units", DataType::Utf8),
            field("i_container", DataType::Utf8),
            field("i_manager_id", DataType::Int64),
            field("i_product_name", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["i_item_id_a", "i_item_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            strings(["i_item_desc_a", "i_item_desc_b"]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            i64s([0, 1]),
            strings(["i_brand_a", "i_brand_b"]),
            i64s([0, 1]),
            strings(["i_class_a", "i_class_b"]),
            i64s([0, 1]),
            strings(["i_category_a", "i_category_b"]),
            i64s([0, 1]),
            strings(["i_manufact_a", "i_manufact_b"]),
            strings(["i_size_a", "i_size_b"]),
            strings(["i_formulation_a", "i_formulation_b"]),
            strings(["i_color_a", "i_color_b"]),
            strings(["i_units_a", "i_units_b"]),
            strings(["i_container_a", "i_container_b"]),
            i64s([0, 1]),
            strings(["i_product_name_a", "i_product_name_b"]),
        ],
    )
}

fn write_promotion(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "promotion",
        schema(vec![
            field("p_promo_sk", DataType::Int64),
            field("p_promo_id", DataType::Utf8),
            field("p_start_date_sk", DataType::Int64),
            field("p_end_date_sk", DataType::Int64),
            field("p_item_sk", DataType::Int64),
            field("p_cost", DataType::Decimal128(15, 2)),
            field("p_response_target", DataType::Int64),
            field("p_promo_name", DataType::Utf8),
            field("p_channel_dmail", DataType::Utf8),
            field("p_channel_email", DataType::Utf8),
            field("p_channel_catalog", DataType::Utf8),
            field("p_channel_tv", DataType::Utf8),
            field("p_channel_radio", DataType::Utf8),
            field("p_channel_press", DataType::Utf8),
            field("p_channel_event", DataType::Utf8),
            field("p_channel_demo", DataType::Utf8),
            field("p_channel_details", DataType::Utf8),
            field("p_purpose", DataType::Utf8),
            field("p_discount_active", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["p_promo_id_a", "p_promo_id_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_15_2([0, 100]),
            i64s([0, 1]),
            strings(["p_promo_name_a", "p_promo_name_b"]),
            strings(["p_channel_dmail_a", "p_channel_dmail_b"]),
            strings(["p_channel_email_a", "p_channel_email_b"]),
            strings(["p_channel_catalog_a", "p_channel_catalog_b"]),
            strings(["p_channel_tv_a", "p_channel_tv_b"]),
            strings(["p_channel_radio_a", "p_channel_radio_b"]),
            strings(["p_channel_press_a", "p_channel_press_b"]),
            strings(["p_channel_event_a", "p_channel_event_b"]),
            strings(["p_channel_demo_a", "p_channel_demo_b"]),
            strings(["p_channel_details_a", "p_channel_details_b"]),
            strings(["p_purpose_a", "p_purpose_b"]),
            strings(["p_discount_active_a", "p_discount_active_b"]),
        ],
    )
}

fn write_reason(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "reason",
        schema(vec![
            field("r_reason_sk", DataType::Int64),
            field("r_reason_id", DataType::Utf8),
            field("r_reason_desc", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["r_reason_id_a", "r_reason_id_b"]),
            strings(["r_reason_desc_a", "r_reason_desc_b"]),
        ],
    )
}

fn write_ship_mode(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "ship_mode",
        schema(vec![
            field("sm_ship_mode_sk", DataType::Int64),
            field("sm_ship_mode_id", DataType::Utf8),
            field("sm_type", DataType::Utf8),
            field("sm_code", DataType::Utf8),
            field("sm_carrier", DataType::Utf8),
            field("sm_contract", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["sm_ship_mode_id_a", "sm_ship_mode_id_b"]),
            strings(["sm_type_a", "sm_type_b"]),
            strings(["sm_code_a", "sm_code_b"]),
            strings(["sm_carrier_a", "sm_carrier_b"]),
            strings(["sm_contract_a", "sm_contract_b"]),
        ],
    )
}

fn write_store(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "store",
        schema(vec![
            field("s_store_sk", DataType::Int64),
            field("s_store_id", DataType::Utf8),
            field("s_rec_start_date", DataType::Date32),
            field("s_rec_end_date", DataType::Date32),
            field("s_closed_date_sk", DataType::Int64),
            field("s_store_name", DataType::Utf8),
            field("s_number_employees", DataType::Int64),
            field("s_floor_space", DataType::Int64),
            field("s_hours", DataType::Utf8),
            field("s_manager", DataType::Utf8),
            field("s_market_id", DataType::Int64),
            field("s_geography_class", DataType::Utf8),
            field("s_market_desc", DataType::Utf8),
            field("s_market_manager", DataType::Utf8),
            field("s_division_id", DataType::Int64),
            field("s_division_name", DataType::Utf8),
            field("s_company_id", DataType::Int64),
            field("s_company_name", DataType::Utf8),
            field("s_street_number", DataType::Utf8),
            field("s_street_name", DataType::Utf8),
            field("s_street_type", DataType::Utf8),
            field("s_suite_number", DataType::Utf8),
            field("s_city", DataType::Utf8),
            field("s_county", DataType::Utf8),
            field("s_state", DataType::Utf8),
            field("s_zip", DataType::Utf8),
            field("s_country", DataType::Utf8),
            field("s_gmt_offset", DataType::Decimal128(5, 2)),
            field("s_tax_percentage", DataType::Decimal128(5, 2)),
        ]),
        vec![
            i64s([0, 1]),
            strings(["s_store_id_a", "s_store_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            i64s([0, 1]),
            strings(["s_store_name_a", "s_store_name_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["s_hours_a", "s_hours_b"]),
            strings(["s_manager_a", "s_manager_b"]),
            i64s([0, 1]),
            strings(["s_geography_class_a", "s_geography_class_b"]),
            strings(["s_market_desc_a", "s_market_desc_b"]),
            strings(["s_market_manager_a", "s_market_manager_b"]),
            i64s([0, 1]),
            strings(["s_division_name_a", "s_division_name_b"]),
            i64s([0, 1]),
            strings(["s_company_name_a", "s_company_name_b"]),
            strings(["s_street_number_a", "s_street_number_b"]),
            strings(["s_street_name_a", "s_street_name_b"]),
            strings(["s_street_type_a", "s_street_type_b"]),
            strings(["s_suite_number_a", "s_suite_number_b"]),
            strings(["s_city_a", "s_city_b"]),
            strings(["s_county_a", "s_county_b"]),
            strings(["s_state_a", "s_state_b"]),
            strings(["s_zip_a", "s_zip_b"]),
            strings(["s_country_a", "s_country_b"]),
            decimals_5_2([0, 100]),
            decimals_5_2([0, 100]),
        ],
    )
}

fn write_store_returns(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "store_returns",
        schema(vec![
            field("sr_returned_date_sk", DataType::Int64),
            field("sr_return_time_sk", DataType::Int64),
            field("sr_item_sk", DataType::Int64),
            field("sr_customer_sk", DataType::Int64),
            field("sr_cdemo_sk", DataType::Int64),
            field("sr_hdemo_sk", DataType::Int64),
            field("sr_addr_sk", DataType::Int64),
            field("sr_store_sk", DataType::Int64),
            field("sr_reason_sk", DataType::Int64),
            field("sr_ticket_number", DataType::Int64),
            field("sr_return_quantity", DataType::Int64),
            field("sr_return_amt", DataType::Decimal128(7, 2)),
            field("sr_return_tax", DataType::Decimal128(7, 2)),
            field("sr_return_amt_inc_tax", DataType::Decimal128(7, 2)),
            field("sr_fee", DataType::Decimal128(7, 2)),
            field("sr_return_ship_cost", DataType::Decimal128(7, 2)),
            field("sr_refunded_cash", DataType::Decimal128(7, 2)),
            field("sr_reversed_charge", DataType::Decimal128(7, 2)),
            field("sr_store_credit", DataType::Decimal128(7, 2)),
            field("sr_net_loss", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_store_sales(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "store_sales",
        schema(vec![
            field("ss_sold_date_sk", DataType::Int64),
            field("ss_sold_time_sk", DataType::Int64),
            field("ss_item_sk", DataType::Int64),
            field("ss_customer_sk", DataType::Int64),
            field("ss_cdemo_sk", DataType::Int64),
            field("ss_hdemo_sk", DataType::Int64),
            field("ss_addr_sk", DataType::Int64),
            field("ss_store_sk", DataType::Int64),
            field("ss_promo_sk", DataType::Int64),
            field("ss_ticket_number", DataType::Int64),
            field("ss_quantity", DataType::Int64),
            field("ss_wholesale_cost", DataType::Decimal128(7, 2)),
            field("ss_list_price", DataType::Decimal128(7, 2)),
            field("ss_sales_price", DataType::Decimal128(7, 2)),
            field("ss_ext_discount_amt", DataType::Decimal128(7, 2)),
            field("ss_ext_sales_price", DataType::Decimal128(7, 2)),
            field("ss_ext_wholesale_cost", DataType::Decimal128(7, 2)),
            field("ss_ext_list_price", DataType::Decimal128(7, 2)),
            field("ss_ext_tax", DataType::Decimal128(7, 2)),
            field("ss_coupon_amt", DataType::Decimal128(7, 2)),
            field("ss_net_paid", DataType::Decimal128(7, 2)),
            field("ss_net_paid_inc_tax", DataType::Decimal128(7, 2)),
            field("ss_net_profit", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_time_dim(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "time_dim",
        schema(vec![
            field("t_time_sk", DataType::Int64),
            field("t_time_id", DataType::Utf8),
            field("t_time", DataType::Int64),
            field("t_hour", DataType::Int64),
            field("t_minute", DataType::Int64),
            field("t_second", DataType::Int64),
            field("t_am_pm", DataType::Utf8),
            field("t_shift", DataType::Utf8),
            field("t_sub_shift", DataType::Utf8),
            field("t_meal_time", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1]),
            strings(["t_time_id_a", "t_time_id_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["t_am_pm_a", "t_am_pm_b"]),
            strings(["t_shift_a", "t_shift_b"]),
            strings(["t_sub_shift_a", "t_sub_shift_b"]),
            strings(["t_meal_time_a", "t_meal_time_b"]),
        ],
    )
}

fn write_warehouse(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "warehouse",
        schema(vec![
            field("w_warehouse_sk", DataType::Int64),
            field("w_warehouse_id", DataType::Utf8),
            field("w_warehouse_name", DataType::Utf8),
            field("w_warehouse_sq_ft", DataType::Int64),
            field("w_street_number", DataType::Utf8),
            field("w_street_name", DataType::Utf8),
            field("w_street_type", DataType::Utf8),
            field("w_suite_number", DataType::Utf8),
            field("w_city", DataType::Utf8),
            field("w_county", DataType::Utf8),
            field("w_state", DataType::Utf8),
            field("w_zip", DataType::Utf8),
            field("w_country", DataType::Utf8),
            field("w_gmt_offset", DataType::Decimal128(5, 2)),
        ]),
        vec![
            i64s([0, 1]),
            strings(["w_warehouse_id_a", "w_warehouse_id_b"]),
            strings(["w_warehouse_name_a", "w_warehouse_name_b"]),
            i64s([0, 1]),
            strings(["w_street_number_a", "w_street_number_b"]),
            strings(["w_street_name_a", "w_street_name_b"]),
            strings(["w_street_type_a", "w_street_type_b"]),
            strings(["w_suite_number_a", "w_suite_number_b"]),
            strings(["w_city_a", "w_city_b"]),
            strings(["w_county_a", "w_county_b"]),
            strings(["w_state_a", "w_state_b"]),
            strings(["w_zip_a", "w_zip_b"]),
            strings(["w_country_a", "w_country_b"]),
            decimals_5_2([0, 100]),
        ],
    )
}

fn write_web_page(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "web_page",
        schema(vec![
            field("wp_web_page_sk", DataType::Int64),
            field("wp_web_page_id", DataType::Utf8),
            field("wp_rec_start_date", DataType::Date32),
            field("wp_rec_end_date", DataType::Date32),
            field("wp_creation_date_sk", DataType::Int64),
            field("wp_access_date_sk", DataType::Int64),
            field("wp_autogen_flag", DataType::Utf8),
            field("wp_customer_sk", DataType::Int64),
            field("wp_url", DataType::Utf8),
            field("wp_type", DataType::Utf8),
            field("wp_char_count", DataType::Int64),
            field("wp_link_count", DataType::Int64),
            field("wp_image_count", DataType::Int64),
            field("wp_max_ad_count", DataType::Int32),
        ]),
        vec![
            i64s([0, 1]),
            strings(["wp_web_page_id_a", "wp_web_page_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["wp_autogen_flag_a", "wp_autogen_flag_b"]),
            i64s([0, 1]),
            strings(["wp_url_a", "wp_url_b"]),
            strings(["wp_type_a", "wp_type_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i32s([0, 1]),
        ],
    )
}

fn write_web_returns(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "web_returns",
        schema(vec![
            field("wr_returned_date_sk", DataType::Int64),
            field("wr_returned_time_sk", DataType::Int64),
            field("wr_item_sk", DataType::Int64),
            field("wr_refunded_customer_sk", DataType::Int64),
            field("wr_refunded_cdemo_sk", DataType::Int64),
            field("wr_refunded_hdemo_sk", DataType::Int64),
            field("wr_refunded_addr_sk", DataType::Int64),
            field("wr_returning_customer_sk", DataType::Int64),
            field("wr_returning_cdemo_sk", DataType::Int64),
            field("wr_returning_hdemo_sk", DataType::Int64),
            field("wr_returning_addr_sk", DataType::Int64),
            field("wr_web_page_sk", DataType::Int64),
            field("wr_reason_sk", DataType::Int64),
            field("wr_order_number", DataType::Int64),
            field("wr_return_quantity", DataType::Int64),
            field("wr_return_amt", DataType::Decimal128(7, 2)),
            field("wr_return_tax", DataType::Decimal128(7, 2)),
            field("wr_return_amt_inc_tax", DataType::Decimal128(7, 2)),
            field("wr_fee", DataType::Decimal128(7, 2)),
            field("wr_return_ship_cost", DataType::Decimal128(7, 2)),
            field("wr_refunded_cash", DataType::Decimal128(7, 2)),
            field("wr_reversed_charge", DataType::Decimal128(7, 2)),
            field("wr_account_credit", DataType::Decimal128(7, 2)),
            field("wr_net_loss", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_web_sales(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "web_sales",
        schema(vec![
            field("ws_sold_date_sk", DataType::Int64),
            field("ws_sold_time_sk", DataType::Int64),
            field("ws_ship_date_sk", DataType::Int64),
            field("ws_item_sk", DataType::Int64),
            field("ws_bill_customer_sk", DataType::Int64),
            field("ws_bill_cdemo_sk", DataType::Int64),
            field("ws_bill_hdemo_sk", DataType::Int64),
            field("ws_bill_addr_sk", DataType::Int64),
            field("ws_ship_customer_sk", DataType::Int64),
            field("ws_ship_cdemo_sk", DataType::Int64),
            field("ws_ship_hdemo_sk", DataType::Int64),
            field("ws_ship_addr_sk", DataType::Int64),
            field("ws_web_page_sk", DataType::Int64),
            field("ws_web_site_sk", DataType::Int64),
            field("ws_ship_mode_sk", DataType::Int64),
            field("ws_warehouse_sk", DataType::Int64),
            field("ws_promo_sk", DataType::Int64),
            field("ws_order_number", DataType::Int64),
            field("ws_quantity", DataType::Int64),
            field("ws_wholesale_cost", DataType::Decimal128(7, 2)),
            field("ws_list_price", DataType::Decimal128(7, 2)),
            field("ws_sales_price", DataType::Decimal128(7, 2)),
            field("ws_ext_discount_amt", DataType::Decimal128(7, 2)),
            field("ws_ext_sales_price", DataType::Decimal128(7, 2)),
            field("ws_ext_wholesale_cost", DataType::Decimal128(7, 2)),
            field("ws_ext_list_price", DataType::Decimal128(7, 2)),
            field("ws_ext_tax", DataType::Decimal128(7, 2)),
            field("ws_coupon_amt", DataType::Decimal128(7, 2)),
            field("ws_ext_ship_cost", DataType::Decimal128(7, 2)),
            field("ws_net_paid", DataType::Decimal128(7, 2)),
            field("ws_net_paid_inc_tax", DataType::Decimal128(7, 2)),
            field("ws_net_paid_inc_ship", DataType::Decimal128(7, 2)),
            field("ws_net_paid_inc_ship_tax", DataType::Decimal128(7, 2)),
            field("ws_net_profit", DataType::Decimal128(7, 2)),
        ]),
        vec![
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            i64s([0, 1]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
            decimals_7_2([0, 100]),
        ],
    )
}

fn write_web_site(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "web_site",
        schema(vec![
            field("web_site_sk", DataType::Int64),
            field("web_site_id", DataType::Utf8),
            field("web_rec_start_date", DataType::Date32),
            field("web_rec_end_date", DataType::Date32),
            field("web_name", DataType::Utf8),
            field("web_open_date_sk", DataType::Int64),
            field("web_close_date_sk", DataType::Int64),
            field("web_class", DataType::Utf8),
            field("web_manager", DataType::Utf8),
            field("web_mkt_id", DataType::Int64),
            field("web_mkt_class", DataType::Utf8),
            field("web_mkt_desc", DataType::Utf8),
            field("web_market_manager", DataType::Utf8),
            field("web_company_id", DataType::Int64),
            field("web_company_name", DataType::Utf8),
            field("web_street_number", DataType::Utf8),
            field("web_street_name", DataType::Utf8),
            field("web_street_type", DataType::Utf8),
            field("web_suite_number", DataType::Utf8),
            field("web_city", DataType::Utf8),
            field("web_county", DataType::Utf8),
            field("web_state", DataType::Utf8),
            field("web_zip", DataType::Utf8),
            field("web_country", DataType::Utf8),
            field("web_gmt_offset", DataType::Decimal128(5, 2)),
            field("web_tax_percentage", DataType::Decimal128(5, 2)),
        ]),
        vec![
            i64s([0, 1]),
            strings(["web_site_id_a", "web_site_id_b"]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            dates([(2000, 1, 1), (2000, 1, 2)]),
            strings(["web_name_a", "web_name_b"]),
            i64s([0, 1]),
            i64s([0, 1]),
            strings(["web_class_a", "web_class_b"]),
            strings(["web_manager_a", "web_manager_b"]),
            i64s([0, 1]),
            strings(["web_mkt_class_a", "web_mkt_class_b"]),
            strings(["web_mkt_desc_a", "web_mkt_desc_b"]),
            strings(["web_market_manager_a", "web_market_manager_b"]),
            i64s([0, 1]),
            strings(["web_company_name_a", "web_company_name_b"]),
            strings(["web_street_number_a", "web_street_number_b"]),
            strings(["web_street_name_a", "web_street_name_b"]),
            strings(["web_street_type_a", "web_street_type_b"]),
            strings(["web_suite_number_a", "web_suite_number_b"]),
            strings(["web_city_a", "web_city_b"]),
            strings(["web_county_a", "web_county_b"]),
            strings(["web_state_a", "web_state_b"]),
            strings(["web_zip_a", "web_zip_b"]),
            strings(["web_country_a", "web_country_b"]),
            decimals_5_2([0, 100]),
            decimals_5_2([0, 100]),
        ],
    )
}

fn write_table(root: &Path, table: &str, schema: Arc<Schema>, columns: Vec<ArrayRef>) -> WorkloadResult<()> {
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let file = File::create(root.join(format!("{table}.parquet")))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn field(name: &'static str, data_type: DataType) -> Field {
    Field::new(name, data_type, false)
}

fn i32s<const N: usize>(values: [i32; N]) -> ArrayRef {
    Arc::new(Int32Array::from(Vec::from(values)))
}

fn i64s<const N: usize>(values: [i64; N]) -> ArrayRef {
    Arc::new(Int64Array::from(Vec::from(values)))
}

fn strings<const N: usize>(values: [&'static str; N]) -> ArrayRef {
    Arc::new(StringArray::from(Vec::from(values)))
}

fn dates<const N: usize>(values: [(i32, u32, u32); N]) -> ArrayRef {
    Arc::new(Date32Array::from(values.map(|(year, month, day)| date32(year, month, day)).to_vec()))
}

fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

fn decimals_5_2<const N: usize>(values: [i128; N]) -> ArrayRef {
    Arc::new(Decimal128Array::from_iter_values(values).with_precision_and_scale(5, 2).unwrap())
}

fn decimals_7_2<const N: usize>(values: [i128; N]) -> ArrayRef {
    Arc::new(Decimal128Array::from_iter_values(values).with_precision_and_scale(7, 2).unwrap())
}

fn decimals_15_2<const N: usize>(values: [i128; N]) -> ArrayRef {
    Arc::new(Decimal128Array::from_iter_values(values).with_precision_and_scale(15, 2).unwrap())
}
