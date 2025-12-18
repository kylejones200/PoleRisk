# Blessed Interface Specification

## The One True Way™

To eliminate confusion and provide **clear guidance for operators, developers, and buyers**, we define **exactly three blessed entry points**. Everything else is internal implementation.

---

## 🖥️ **CLI: `pole-health`**
**Target Users:** Operations teams, DevOps, batch processing  
**Use Cases:** Daily operations, automation, emergency response

### **Core Commands**
```bash
# ASSESSMENT: Generate risk reports for pole fleets
pole-health assess \
  --fleet data/poles.csv \
  --soil data/soil-samples.csv \
  --output reports/ \
  --format detailed

# SCHEDULING: Create maintenance work orders  
pole-health schedule \
  --budget 500000 \
  --timeframe "2024-Q2" \
  --region "North District" \
  --output maintenance-plan.csv

# STORM-PREP: Emergency storm hardening
pole-health storm-prep \
  --region "Hurricane Zone" \
  --severity high \
  --timeline "48-hours" \
  --export-work-orders

# MONITORING: Health status checks
pole-health status \
  --fleet data/poles.csv \
  --alerts-only \
  --threshold critical
```

### **Configuration**
```bash
# Global config file: ~/.pole-health/config.yaml
pole-health config set database.url "postgresql://..."
pole-health config set weather.api_key "..."
pole-health config show
```

---

## 🌐 **API: `/api/v1/`**
**Target Users:** Enterprise integrations, custom applications  
**Use Cases:** GIS integration, CMMS connectivity, real-time monitoring

### **Core Endpoints**

#### **Health Assessment**
```http
GET /api/v1/poles/{pole_id}/health
GET /api/v1/poles/health?region=north&limit=1000
POST /api/v1/assessments/batch
```

#### **Maintenance Management**
```http
GET /api/v1/maintenance/recommendations?priority=critical
POST /api/v1/maintenance/schedule
GET /api/v1/maintenance/workorders/{order_id}
```

#### **Real-time Monitoring**
```http
GET /api/v1/alerts/active
POST /api/v1/alerts/acknowledge/{alert_id}
WebSocket: /api/v1/stream/health-updates
```

### **Authentication**
```bash
# API Key (for system integration)
curl -H "X-API-Key: your-api-key" /api/v1/poles/health

# JWT (for user applications)  
curl -H "Authorization: Bearer jwt-token" /api/v1/poles/health
```

---

## 📊 **Dashboard: Pole Health Center**
**Target Users:** Executives, managers, analysts  
**Use Cases:** Strategic planning, performance monitoring, decision support

### **Launch Command**
```bash
# Start the unified dashboard server
pole-health dashboard \
  --port 8080 \
  --host 0.0.0.0 \
  --auth enabled

# Opens: http://localhost:8080/health-center
```

### **Dashboard Modules**

#### **Executive Overview**
- Fleet health summary (KPIs and trends)
- Financial impact (cost savings, ROI)
- Risk distribution (geographic and priority)
- Performance metrics (prediction accuracy)

#### **Operations Center**  
- Real-time alerts and notifications
- Work order management and tracking
- Crew dispatch and resource allocation
- Storm preparation dashboard

#### **Analytics Workbench**
- Interactive risk modeling
- "What-if" scenario analysis
- Custom report generation
- Data export and API testing

---

## Migration Commands

To help users transition from the old multi-entry system:

```bash
# Replace old commands with new blessed interface
pole-health migrate check        # Scan for deprecated usage
pole-health migrate convert      # Auto-convert old scripts
pole-health migrate validate     # Test converted workflows
```

### **Deprecation Warnings**
```bash
$ python main.py --poles data.csv
⚠️  WARNING: main.py is deprecated. Use: pole-health assess --fleet data.csv

$ streamlit run dashboard_app.py  
⚠️  WARNING: Direct Streamlit launch is deprecated. Use: pole-health dashboard

$ python api_server.py
⚠️  WARNING: Direct server launch is deprecated. Use: pole-health server
```

---

## Implementation Strategy

### **Phase 1: Create Blessed CLI** ✅
```bash
# New unified command structure
pip install pole-health-assessment
pole-health --version  # v1.0.0
pole-health --help     # Show all blessed commands
```

### **Phase 2: Wrap Existing Functionality** 
```python
# pole_health/cli/assess.py
def assess_command(fleet_file, soil_file, output_dir):
    """Blessed assessment command - wraps existing main.py logic"""
    from ..legacy.main import assess_pole_fleet  # Internal import
    return assess_pole_fleet(...)

# pole_health/cli/dashboard.py  
def dashboard_command(port, host):
    """Blessed dashboard command - wraps existing dashboard_app.py"""
    from ..legacy.dashboard_app import PoleDashboard  # Internal import
    return PoleDashboard().run(port=port, host=host)
```

### **Phase 3: Deprecate Old Entry Points**
```python
# main.py (deprecated)
import warnings
warnings.warn(
    "main.py is deprecated. Use 'pole-health assess' instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### **Phase 4: Remove Legacy Code** (v2.0)
- Delete standalone scripts
- Clean up internal APIs  
- Consolidate documentation

---

## Benefits of This Approach

### **For Operators**
✅ **One command to learn:** `pole-health <action>`  
✅ **Consistent interface:** Same flags and options  
✅ **Clear documentation:** Single help system  
✅ **Easy automation:** Scriptable and pipeable

### **For Integrators**  
✅ **One API to integrate:** `/api/v1/`  
✅ **Stable interface:** Versioned and backward-compatible  
✅ **Standard auth:** JWT + API keys  
✅ **Complete functionality:** No hidden internal APIs

### **For Executives**
✅ **One dashboard URL:** Bookmark and go  
✅ **Unified experience:** All features in one place  
✅ **Role-based access:** Executive vs operator views  
✅ **Professional appearance:** No debug screens or research tools

### **For Developers**
✅ **Clear architecture:** Public vs internal APIs  
✅ **Easy testing:** Mock blessed interfaces only  
✅ **Simple deployment:** One binary, one container  
✅ **Maintainable code:** Internal refactoring without breaking users

---

*This specification defines the public contract. Everything not listed here is internal implementation subject to change.*
